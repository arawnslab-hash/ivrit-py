"""
Test batch transcription functionality using the asimov.mp3 file.

Batch mode is selected by passing a list (even a one-element list) to one of
path/url/blob. This test covers:
1. Sync batch non-streaming -> List[dict] in input order
2. One-element list -> 1-element List[dict] (not a bare dict)
3. Sync batch streaming -> (index, Segment) tuples with monotonic indices
4. Async batch streaming -> (index, Segment) tuples
5. Per-item error isolation
6. Source validation (mixed kinds, empty list)

Local-engine cases use faster-whisper; RunPod cases require
RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID.
"""

import os
import pytest
from pathlib import Path

from ivrit import load_model
from ivrit.audio import TranscriptionModel
from ivrit.types import Segment, Word
from ivrit.utils import emit_progress

LOCAL_ENGINE = "faster-whisper"
LOCAL_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"


class _FakeModel(TranscriptionModel):
    """Minimal in-memory model that emits one progress event (with an
    engine-specific extra) and yields one segment per source. Lets us exercise
    batch orchestration and progress wrapping without a heavy decode."""

    def __init__(self):
        super().__init__(engine="fake", model="fake")

    def transcribe_core(self, *, path=None, url=None, blob=None, language=None,
                        diarize=False, diarization_args=None, output_options,
                        verbose=False, on_progress=None, **kwargs):
        emit_progress(on_progress, phase="transcription", step="decode",
                      step_fraction=1.0, description="decoding", engine_key="v")
        yield Segment(text=(path or url or blob or ""), start=0.0, end=1.0)


class TestBatchTranscription:
    """Test batch transcription functionality using asimov.mp3"""

    @pytest.fixture
    def audio_file_path(self):
        """Get the path to the asimov.mp3 test file"""
        test_dir = Path(__file__).parent
        audio_path = test_dir / "asimov.mp3"
        assert audio_path.exists(), f"Test audio file not found: {audio_path}"
        return str(audio_path)

    @pytest.fixture
    def local_model(self):
        """Load a local transcription model for batch tests"""
        return load_model(engine=LOCAL_ENGINE, model=LOCAL_MODEL)

    def _assert_valid_segment(self, segment):
        assert isinstance(segment, Segment), f"Not a Segment: {type(segment)}"
        assert isinstance(segment.words, list), f"Words not a list: {type(segment.words)}"
        for word in segment.words:
            assert isinstance(word, Word), f"Not a Word: {type(word)}"

    def _assert_valid_result_dict(self, result):
        assert isinstance(result, dict), f"Result is not a dict: {type(result)}"
        assert 'segments' in result, "Result dict missing 'segments' key"
        assert len(result['segments']) > 0, "No segments in result"
        for segment in result['segments']:
            self._assert_valid_segment(segment)

    def test_batch_non_streaming(self, local_model, audio_file_path):
        """Sync batch non-streaming returns List[dict] of matching length and order"""
        results = local_model.transcribe(
            path=[audio_file_path, audio_file_path],
            language='he',
            stream=False,
        )
        assert isinstance(results, list), f"Batch result is not a list: {type(results)}"
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        for result in results:
            self._assert_valid_result_dict(result)

    def test_one_element_list_returns_list(self, local_model, audio_file_path):
        """A one-element list returns a 1-element List[dict], not a bare dict"""
        results = local_model.transcribe(
            path=[audio_file_path],
            language='he',
            stream=False,
        )
        assert isinstance(results, list), f"One-element batch is not a list: {type(results)}"
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        self._assert_valid_result_dict(results[0])

    def test_batch_streaming(self, local_model, audio_file_path):
        """Sync batch streaming yields (index, Segment) tuples with monotonic indices"""
        results = list(local_model.transcribe(
            path=[audio_file_path, audio_file_path],
            language='he',
            stream=True,
        ))
        assert len(results) > 0, "No items yielded from batch streaming"

        last_index = -1
        seen_indices = set()
        for item in results:
            assert isinstance(item, tuple) and len(item) == 2, f"Item is not a 2-tuple: {item!r}"
            index, payload = item
            assert isinstance(index, int), f"Index is not an int: {type(index)}"
            assert index >= last_index, "Indices are not monotonic non-decreasing"
            last_index = index
            seen_indices.add(index)
            self._assert_valid_segment(payload)

        assert seen_indices == {0, 1}, f"Expected indices {{0, 1}}, got {seen_indices}"

    @pytest.mark.asyncio
    async def test_async_batch_streaming(self, local_model, audio_file_path):
        """Async batch streaming yields (index, Segment) tuples"""
        items = []
        async for item in local_model.transcribe_async(
            path=[audio_file_path, audio_file_path],
            language='he',
        ):
            items.append(item)

        assert len(items) > 0, "No items yielded from async batch streaming"

        seen_indices = set()
        for item in items:
            assert isinstance(item, tuple) and len(item) == 2, f"Item is not a 2-tuple: {item!r}"
            index, payload = item
            assert isinstance(index, int), f"Index is not an int: {type(index)}"
            seen_indices.add(index)
            self._assert_valid_segment(payload)

        assert seen_indices == {0, 1}, f"Expected indices {{0, 1}}, got {seen_indices}"

    def test_error_isolation_non_streaming(self, local_model, audio_file_path):
        """A failed input is isolated; valid inputs still produce results"""
        missing = "/does/not/exist.mp3"
        results = local_model.transcribe(
            path=[audio_file_path, missing],
            language='he',
            stream=False,
        )
        assert isinstance(results, list)
        assert len(results) == 2

        # First (valid) input produced a well-formed result
        self._assert_valid_result_dict(results[0])

        # Second (missing) input produced an isolated error entry
        error_entry = results[1]
        assert isinstance(error_entry, dict)
        assert "error" in error_entry, f"Missing 'error' key: {error_entry}"
        assert error_entry["source"] == "path", f"Unexpected source: {error_entry}"
        assert error_entry["input"] == missing, f"Unexpected input: {error_entry}"

    def test_error_isolation_streaming(self, local_model, audio_file_path):
        """Streaming batch yields (index, Exception) for a failed input and good results otherwise"""
        missing = "/does/not/exist.mp3"
        results = list(local_model.transcribe(
            path=[audio_file_path, missing],
            language='he',
            stream=True,
        ))

        good = [(i, p) for (i, p) in results if isinstance(p, Segment)]
        errors = [(i, p) for (i, p) in results if isinstance(p, Exception)]

        assert len(good) > 0, "No valid segments yielded"
        assert all(i == 0 for i, _ in good), "Valid segments not attributed to index 0"
        assert len(errors) == 1, f"Expected exactly one error item, got {len(errors)}"
        assert errors[0][0] == 1, "Error not attributed to index 1"

    def test_validation_mixed_sources(self, local_model, audio_file_path):
        """Providing two source kinds raises ValueError"""
        with pytest.raises(ValueError):
            local_model.transcribe(
                path=[audio_file_path],
                url=["https://example.com/a.mp3"],
                language='he',
            )

    def test_validation_empty_list(self, local_model):
        """An empty source list raises ValueError"""
        with pytest.raises(ValueError):
            local_model.transcribe(path=[], language='he')

    def test_batch_progress_attribution(self):
        """Batch mode injects batch_index/batch_total into extra without clobbering engine extras"""
        model = _FakeModel()
        events = []
        list(model.transcribe(path=["a", "b"], on_progress=events.append, stream=True))

        assert len(events) == 2, f"Expected one event per item, got {len(events)}"
        for ev in events:
            assert ev["extra"]["engine_key"] == "v", "Engine-supplied extra was clobbered"
            assert ev["extra"]["batch_total"] == 2, f"Wrong batch_total: {ev['extra']}"
        assert {ev["extra"]["batch_index"] for ev in events} == {0, 1}, "Wrong batch_index attribution"

    def test_single_progress_not_wrapped(self):
        """Single (bare-string) calls pass the callback through with no batch keys"""
        model = _FakeModel()
        events = []
        list(model.transcribe(path="a", on_progress=events.append, stream=True))

        assert len(events) == 1
        assert events[0]["extra"]["engine_key"] == "v"
        assert "batch_index" not in events[0]["extra"], "batch_index leaked into single-mode event"
        assert "batch_total" not in events[0]["extra"], "batch_total leaked into single-mode event"

    def test_runpod_batch_non_streaming(self, audio_file_path):
        """RunPod sync batch non-streaming returns List[dict]"""
        api_key = os.getenv("RUNPOD_API_KEY")
        endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")

        assert api_key, "RUNPOD_API_KEY environment variable is required for RunPod tests"
        assert endpoint_id, "RUNPOD_ENDPOINT_ID environment variable is required for RunPod tests"

        model = load_model(
            engine="runpod",
            model=LOCAL_MODEL,
            api_key=api_key,
            endpoint_id=endpoint_id,
        )
        results = model.transcribe(
            path=[audio_file_path, audio_file_path],
            language='he',
            stream=False,
        )
        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            self._assert_valid_result_dict(result)

    @pytest.mark.asyncio
    async def test_runpod_async_batch_streaming(self, audio_file_path):
        """RunPod async batch streaming yields (index, Segment) tuples"""
        api_key = os.getenv("RUNPOD_API_KEY")
        endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")

        assert api_key, "RUNPOD_API_KEY environment variable is required for RunPod tests"
        assert endpoint_id, "RUNPOD_ENDPOINT_ID environment variable is required for RunPod tests"

        model = load_model(
            engine="runpod",
            model=LOCAL_MODEL,
            api_key=api_key,
            endpoint_id=endpoint_id,
        )
        seen_indices = set()
        async for item in model.transcribe_async(
            path=[audio_file_path, audio_file_path],
            language='he',
        ):
            assert isinstance(item, tuple) and len(item) == 2
            index, payload = item
            seen_indices.add(index)
            self._assert_valid_segment(payload)

        assert seen_indices == {0, 1}, f"Expected indices {{0, 1}}, got {seen_indices}"
