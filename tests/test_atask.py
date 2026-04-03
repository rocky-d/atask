import asyncio as aio
from contextvars import Context, ContextVar
from typing import Any

import pytest

import atask
from atask import AsyncTask, AsyncTaskGroup

# ---------------------------------------------------------------------------
# Concrete helpers
# ---------------------------------------------------------------------------


class SimpleTask(AsyncTask[int]):
    async def _run(self) -> int:
        return 42


class SlowTask(AsyncTask[str]):
    def __init__(self, delay: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._delay = delay

    async def _run(self) -> str:
        await aio.sleep(self._delay)
        return "done"


class FailingTask(AsyncTask[None]):
    async def _run(self) -> None:
        raise ValueError("boom")


class BareTask(AsyncTask[int]):
    """Does not override _run – inherits NotImplementedError."""

    pass


class LifecycleTask(AsyncTask[int]):
    """Tracks lifecycle calls for verification."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.start_called = False
        self.stop_called = False
        self.cancel_called = False

    async def _run(self) -> int:
        await aio.sleep(0.05)
        return 1

    async def start(self) -> None:
        if self.started:
            return
        self.start_called = True
        await super().start()

    async def cancel(self, msg: Any | None = None) -> None:
        if not self.started:
            return
        self.cancel_called = True
        await super().cancel(msg)

    async def stop(self, exc_type=None, exc_value=None, exc_traceback=None) -> None:
        if not self.started:
            return
        if not self.done:
            raise aio.InvalidStateError() from exc_value
        self.stop_called = True
        await super().stop(exc_type, exc_value, exc_traceback)


# ===========================================================================
# AsyncTask tests
# ===========================================================================


class TestAsyncTaskInit:
    async def test_initial_state(self) -> None:
        task = SimpleTask()
        assert task.started is False
        assert task.done is True
        assert task.cancelled is False
        assert task.running is False
        assert task.name is None
        assert task.context is None

    async def test_initial_result_raises(self) -> None:
        task = SimpleTask()
        with pytest.raises(aio.InvalidStateError):
            _ = task.result

    async def test_initial_exception(self) -> None:
        task = SimpleTask()
        exc = task.exception
        assert isinstance(exc, aio.InvalidStateError)

    async def test_name_and_context(self) -> None:
        ctx = Context()
        task = SimpleTask(name="my-task", context=ctx)
        assert task.name == "my-task"
        assert task.context is ctx


class TestAsyncTaskStart:
    async def test_start_sets_started(self) -> None:
        task = SimpleTask()
        await task.start()
        assert task.started is True
        await task.join()
        await task.stop()

    async def test_start_idempotent(self) -> None:
        task = SimpleTask()
        await task.start()
        fut1 = task._fut
        await task.start()
        fut2 = task._fut
        assert fut1 is fut2
        await task.join()
        await task.stop()

    async def test_running_while_executing(self) -> None:
        task = SlowTask(delay=0.2)
        await task.start()
        assert task.running is True
        assert task.done is False
        await task.join()
        assert task.running is False
        assert task.done is True
        await task.stop()


class TestAsyncTaskJoin:
    async def test_join_without_start_is_noop(self) -> None:
        task = SimpleTask()
        await task.join()  # should not raise

    async def test_join_waits_for_result(self) -> None:
        task = SimpleTask()
        await task.start()
        await task.join()
        assert task.done is True
        assert task.result == 42
        await task.stop()


class TestAsyncTaskResult:
    async def test_result_after_completion(self) -> None:
        task = SimpleTask()
        await task.start()
        await task.join()
        assert task.result == 42
        await task.stop()

    async def test_exception_after_failure(self) -> None:
        task = FailingTask()
        await task.start()
        with pytest.raises(ValueError, match="boom"):
            await task.join()
        exc = task.exception
        assert isinstance(exc, ValueError)
        await task.stop()


class TestAsyncTaskCancel:
    async def test_cancel_running_task(self) -> None:
        task = SlowTask(delay=10.0)
        await task.start()
        await task.cancel()
        assert task.cancelled is True
        assert task.done is True
        await task.stop()

    async def test_cancel_with_message(self) -> None:
        task = SlowTask(delay=10.0)
        await task.start()
        await task.cancel(msg="abort")
        assert task.cancelled is True
        await task.stop()

    async def test_cancel_without_start_is_noop(self) -> None:
        task = SimpleTask()
        await task.cancel()  # should not raise
        assert task.started is False


class TestAsyncTaskStop:
    async def test_stop_resets_started(self) -> None:
        task = SimpleTask()
        await task.start()
        await task.join()
        await task.stop()
        assert task.started is False

    async def test_stop_without_start_is_noop(self) -> None:
        task = SimpleTask()
        await task.stop()  # should not raise

    async def test_stop_while_running_raises(self) -> None:
        task = SlowTask(delay=10.0)
        await task.start()
        with pytest.raises(aio.InvalidStateError):
            await task.stop()
        await task.cancel()
        await task.stop()

    async def test_stop_with_exc_info(self) -> None:
        task = SimpleTask()
        await task.start()
        await task.join()
        await task.stop(ValueError, ValueError("test"), None)
        assert task.started is False

    async def test_stop_running_raises_with_exc_value_chain(self) -> None:
        task = SlowTask(delay=10.0)
        await task.start()
        original = ValueError("original")
        with pytest.raises(aio.InvalidStateError) as exc_info:
            await task.stop(ValueError, original, None)
        assert exc_info.value.__cause__ is original
        await task.cancel()
        await task.stop()


class TestAsyncTaskNotImplemented:
    async def test_bare_task_raises_not_implemented(self) -> None:
        task = BareTask()
        await task.start()
        with pytest.raises(NotImplementedError):
            await task.join()
        await task.stop()


class TestAsyncTaskAwait:
    async def test_await_returns_result(self) -> None:
        task = SimpleTask()
        await task.start()
        result = await task
        assert result == 42
        await task.stop()


class TestAsyncTaskContextManager:
    async def test_async_with(self) -> None:
        async with SimpleTask() as task:
            result = await task
            assert result == 42
        assert task.started is False

    async def test_async_with_failing_task(self) -> None:
        with pytest.raises(ValueError, match="boom"):
            async with FailingTask() as task:
                await task

    async def test_async_with_cancel_before_exit(self) -> None:
        async with SlowTask(delay=10.0) as task:
            assert task.started is True
            await task.cancel()
        assert task.started is False

    async def test_context_manager_stops_after_done(self) -> None:
        async with SimpleTask() as task:
            await task.join()
        assert task.started is False
        assert task.done is True


class TestAsyncTaskLifecycleOverride:
    async def test_lifecycle_calls(self) -> None:
        task = LifecycleTask()
        await task.start()
        assert task.start_called is True
        await task.join()
        await task.stop()
        assert task.stop_called is True

    async def test_cancel_override(self) -> None:
        task = LifecycleTask()
        await task.start()
        await task.cancel()
        assert task.cancel_called is True
        await task.stop()

    async def test_start_idempotent(self) -> None:
        task = LifecycleTask()
        await task.start()
        task.start_called = False
        await task.start()
        assert task.start_called is False
        await task.join()
        await task.stop()

    async def test_cancel_without_start_is_noop(self) -> None:
        task = LifecycleTask()
        await task.cancel()
        assert task.cancel_called is False

    async def test_stop_without_start_is_noop(self) -> None:
        task = LifecycleTask()
        await task.stop()
        assert task.stop_called is False

    async def test_stop_while_running_raises(self) -> None:
        task = LifecycleTask()
        await task.start()
        with pytest.raises(aio.InvalidStateError):
            await task.stop()
        await task.cancel()
        await task.stop()


class TestAsyncTaskContextVar:
    async def test_context_propagation(self) -> None:
        var: ContextVar[int] = ContextVar("var")

        class CtxTask(AsyncTask[int]):
            async def _run(self) -> int:
                return var.get()

        ctx = Context()
        ctx.run(var.set, 99)
        task = CtxTask(context=ctx)
        await task.start()
        result = await task
        assert result == 99
        await task.stop()


class TestAsyncTaskName:
    async def test_name_forwarded(self) -> None:
        task = SimpleTask(name="test-name")
        await task.start()
        assert isinstance(task._fut, aio.Task)
        assert task._fut.get_name() == "test-name"
        await task.join()
        await task.stop()


# ===========================================================================
# AsyncTaskGroup tests
# ===========================================================================


class TestAsyncTaskGroupInit:
    async def test_group_initial_state(self) -> None:
        group = AsyncTaskGroup([SimpleTask(), SimpleTask()])
        assert group.started is False
        assert group.done is True

    async def test_empty_group(self) -> None:
        async with AsyncTaskGroup([]) as group:
            result = await group
        assert result == []


class TestAsyncTaskGroupRun:
    async def test_group_collects_results(self) -> None:
        t1 = SimpleTask()
        t2 = SimpleTask()
        async with AsyncTaskGroup([t1, t2]) as group:
            results = await group
        assert results == [42, 42]

    async def test_group_preserves_order(self) -> None:
        class ValueTask(AsyncTask[int]):
            def __init__(self, val: int, delay: float, **kw: Any) -> None:
                super().__init__(**kw)
                self._val = val
                self._delay = delay

            async def _run(self) -> int:
                await aio.sleep(self._delay)
                return self._val

        t1 = ValueTask(1, 0.1)
        t2 = ValueTask(2, 0.01)
        t3 = ValueTask(3, 0.05)
        async with AsyncTaskGroup([t1, t2, t3]) as group:
            results = await group
        assert results == [1, 2, 3]

    async def test_group_with_name(self) -> None:
        group = AsyncTaskGroup([SimpleTask()], name="grp")
        assert group.name == "grp"
        async with group as g:
            await g
        assert g.started is False


class TestAsyncTaskGroupCancel:
    async def test_cancel_group(self) -> None:
        t1 = SlowTask(delay=10.0)
        t2 = SlowTask(delay=10.0)
        group = AsyncTaskGroup([t1, t2])
        await group.start()
        await group.cancel()
        assert t1.cancelled is True
        assert t2.cancelled is True
        assert group.cancelled is True
        await group.stop()

    async def test_cancel_group_without_start(self) -> None:
        group = AsyncTaskGroup([SimpleTask()])
        await group.cancel()  # should not raise
        assert group.started is False


class TestAsyncTaskGroupStop:
    async def test_stop_group(self) -> None:
        t1 = SimpleTask()
        t2 = SimpleTask()
        group = AsyncTaskGroup([t1, t2])
        await group.start()
        await group.join()
        await group.stop()
        assert group.started is False
        assert t1.started is False
        assert t2.started is False

    async def test_stop_group_without_start(self) -> None:
        group = AsyncTaskGroup([SimpleTask()])
        await group.stop()  # should not raise

    async def test_stop_running_group_raises(self) -> None:
        t1 = SlowTask(delay=10.0)
        group = AsyncTaskGroup([t1])
        await group.start()
        with pytest.raises(ExceptionGroup):
            await group.stop()
        await group.cancel()
        await group.stop()


class TestAsyncTaskGroupStart:
    async def test_start_is_idempotent(self) -> None:
        t1 = SimpleTask()
        group = AsyncTaskGroup([t1])
        await group.start()
        fut1 = group._fut
        await group.start()
        fut2 = group._fut
        assert fut1 is fut2
        await group.join()
        await group.stop()

    async def test_start_starts_subtasks(self) -> None:
        t1 = SimpleTask()
        t2 = SimpleTask()
        group = AsyncTaskGroup([t1, t2])
        await group.start()
        assert t1.started is True
        assert t2.started is True
        await group.join()
        await group.stop()


class TestAsyncTaskGroupContextManager:
    async def test_async_with_group(self) -> None:
        t1 = SimpleTask()
        async with AsyncTaskGroup([t1]) as group:
            results = await group
        assert results == [42]
        assert group.started is False

    async def test_async_with_cancel_group(self) -> None:
        t1 = SlowTask(delay=10.0)
        async with AsyncTaskGroup([t1]) as group:
            await group.cancel()
        assert group.started is False


class TestAsyncTaskGroupFailure:
    async def test_subtask_failure_propagates(self) -> None:
        t1 = SimpleTask()
        t2 = FailingTask()
        group = AsyncTaskGroup([t1, t2])
        await group.start()
        with pytest.raises(ExceptionGroup):
            await group.join()
        await group.stop()


class TestModuleExports:
    def test_all_exports(self) -> None:
        assert hasattr(atask, "AsyncTask")
        assert hasattr(atask, "AsyncTaskGroup")
        assert set(atask.__all__) == {"AsyncTask", "AsyncTaskGroup"}
