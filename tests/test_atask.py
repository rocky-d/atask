import asyncio as aio
from contextvars import Context, ContextVar
from typing import Any

import pytest

from atask import AsyncTask, AsyncTaskGroup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleTask(AsyncTask[int]):
    async def _atask(self) -> int:
        return 22


class SlowTask(AsyncTask[str]):
    def __init__(self, delay: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._delay = delay

    async def _atask(self) -> str:
        await aio.sleep(self._delay)
        return "done"


class FailingTask(AsyncTask[None]):
    async def _atask(self) -> None:
        raise ValueError("boom")


class BareTask(AsyncTask[int]):
    """Does not override _atask."""

    pass


class LifecycleTask(AsyncTask[int]):
    """Tracks lifecycle calls for verification."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.astart_called = False
        self.astop_called = False
        self.acancel_called = False

    async def _atask(self) -> int:
        await aio.sleep(0.05)
        return 1

    async def astart(self) -> None:
        if self.started:
            return
        self.astart_called = True
        await super().astart()

    async def acancel(self, msg: Any | None = None) -> None:
        if not self.started:
            return
        self.acancel_called = True
        await super().acancel(msg)

    async def astop(self, exc_type=None, exc_value=None, exc_traceback=None) -> None:
        if not self.started:
            return
        if not self.done:
            raise aio.InvalidStateError() from exc_value
        self.astop_called = True
        await super().astop(exc_type, exc_value, exc_traceback)


# ===========================================================================
# AsyncTask
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


class TestAsyncTaskAstart:
    async def test_astart_sets_started(self) -> None:
        task = SimpleTask()
        await task.astart()
        assert task.started is True
        await task.ajoin()
        await task.astop()

    async def test_astart_idempotent(self) -> None:
        task = SimpleTask()
        await task.astart()
        fut1 = task._fut
        await task.astart()
        fut2 = task._fut
        assert fut1 is fut2
        await task.ajoin()
        await task.astop()

    async def test_running_while_executing(self) -> None:
        task = SlowTask(delay=0.2)
        await task.astart()
        assert task.running is True
        assert task.done is False
        await task.ajoin()
        assert task.running is False
        assert task.done is True
        await task.astop()


class TestAsyncTaskAjoin:
    async def test_ajoin_without_astart_is_noop(self) -> None:
        task = SimpleTask()
        await task.ajoin()  # should not raise

    async def test_ajoin_waits_for_result(self) -> None:
        task = SimpleTask()
        await task.astart()
        await task.ajoin()
        assert task.done is True
        assert task.result == 22
        await task.astop()


class TestAsyncTaskResult:
    async def test_result_after_completion(self) -> None:
        task = SimpleTask()
        await task.astart()
        await task.ajoin()
        assert task.result == 22
        await task.astop()

    async def test_exception_after_failure(self) -> None:
        task = FailingTask()
        await task.astart()
        with pytest.raises(ValueError, match="boom"):
            await task.ajoin()
        exc = task.exception
        assert isinstance(exc, ValueError)
        await task.astop()


class TestAsyncTaskAcancel:
    async def test_acancel_running_task(self) -> None:
        task = SlowTask(delay=10.0)
        await task.astart()
        await task.acancel()
        assert task.cancelled is True
        assert task.done is True
        await task.astop()

    async def test_acancel_with_message(self) -> None:
        task = SlowTask(delay=10.0)
        await task.astart()
        await task.acancel(msg="abort")
        assert task.cancelled is True
        await task.astop()

    async def test_acancel_without_astart_is_noop(self) -> None:
        task = SimpleTask()
        await task.acancel()  # should not raise
        assert task.started is False


class TestAsyncTaskAstop:
    async def test_astop_resets_started(self) -> None:
        task = SimpleTask()
        await task.astart()
        await task.ajoin()
        await task.astop()
        assert task.started is False

    async def test_astop_without_astart_is_noop(self) -> None:
        task = SimpleTask()
        await task.astop()  # should not raise

    async def test_astop_while_running_raises(self) -> None:
        task = SlowTask(delay=10.0)
        await task.astart()
        with pytest.raises(aio.InvalidStateError):
            await task.astop()
        await task.acancel()
        await task.astop()

    async def test_astop_with_exc_info(self) -> None:
        task = SimpleTask()
        await task.astart()
        await task.ajoin()
        await task.astop(ValueError, ValueError("test"), None)
        assert task.started is False

    async def test_astop_running_raises_with_exc_value_chain(self) -> None:
        task = SlowTask(delay=10.0)
        await task.astart()
        original = ValueError("original")
        with pytest.raises(aio.InvalidStateError) as exc_info:
            await task.astop(ValueError, original, None)
        assert exc_info.value.__cause__ is original
        await task.acancel()
        await task.astop()


class TestAsyncTaskNotImplemented:
    async def test_bare_task_raises_not_implemented(self) -> None:
        task = BareTask()
        await task.astart()
        with pytest.raises(NotImplementedError):
            await task.ajoin()
        await task.astop()


class TestAsyncTaskAwait:
    async def test_await_returns_result(self) -> None:
        task = SimpleTask()
        await task.astart()
        result = await task
        assert result == 22
        await task.astop()


class TestAsyncTaskContextManager:
    async def test_async_with(self) -> None:
        async with SimpleTask() as task:
            result = await task
            assert result == 22
        assert task.started is False

    async def test_async_with_failing_task(self) -> None:
        with pytest.raises(ValueError, match="boom"):
            async with FailingTask() as task:
                await task

    async def test_async_with_acancel_before_exit(self) -> None:
        async with SlowTask(delay=10.0) as task:
            assert task.started is True
            await task.acancel()
        assert task.started is False

    async def test_context_manager_astops_after_done(self) -> None:
        async with SimpleTask() as task:
            await task.ajoin()
        assert task.started is False
        assert task.done is True


class TestAsyncTaskLifecycleOverride:
    async def test_lifecycle_calls(self) -> None:
        task = LifecycleTask()
        await task.astart()
        assert task.astart_called is True
        await task.ajoin()
        await task.astop()
        assert task.astop_called is True

    async def test_acancel_override(self) -> None:
        task = LifecycleTask()
        await task.astart()
        await task.acancel()
        assert task.acancel_called is True
        await task.astop()

    async def test_astart_idempotent(self) -> None:
        task = LifecycleTask()
        await task.astart()
        task.astart_called = False
        await task.astart()
        assert task.astart_called is False
        await task.ajoin()
        await task.astop()

    async def test_acancel_without_astart_is_noop(self) -> None:
        task = LifecycleTask()
        await task.acancel()
        assert task.acancel_called is False

    async def test_astop_without_astart_is_noop(self) -> None:
        task = LifecycleTask()
        await task.astop()
        assert task.astop_called is False

    async def test_astop_while_running_raises(self) -> None:
        task = LifecycleTask()
        await task.astart()
        with pytest.raises(aio.InvalidStateError):
            await task.astop()
        await task.acancel()
        await task.astop()


class TestAsyncTaskContextVar:
    async def test_context_propagation(self) -> None:
        var = ContextVar("var")

        class CtxTask(AsyncTask[int]):
            async def _atask(self) -> int:
                return var.get()

        ctx = Context()
        ctx.run(var.set, 99)
        task = CtxTask(context=ctx)
        await task.astart()
        result = await task
        assert result == 99
        await task.astop()


class TestAsyncTaskName:
    async def test_name_forwarded(self) -> None:
        task = SimpleTask(name="test-name")
        await task.astart()
        assert isinstance(task._fut, aio.Task)
        assert task._fut.get_name() == "test-name"
        await task.ajoin()
        await task.astop()


# ===========================================================================
# AsyncTaskGroup
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
        assert results == [22, 22]

    async def test_group_preserves_order(self) -> None:
        class ValueTask(AsyncTask[int]):
            def __init__(self, val: int, delay: float, **kw: Any) -> None:
                super().__init__(**kw)
                self._val = val
                self._delay = delay

            async def _atask(self) -> int:
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


class TestAsyncTaskGroupAcancel:
    async def test_acancel_group(self) -> None:
        t1 = SlowTask(delay=10.0)
        t2 = SlowTask(delay=10.0)
        group = AsyncTaskGroup([t1, t2])
        await group.astart()
        await group.acancel()
        assert t1.cancelled is True
        assert t2.cancelled is True
        assert group.cancelled is True
        await group.astop()

    async def test_acancel_group_without_astart(self) -> None:
        group = AsyncTaskGroup([SimpleTask()])
        await group.acancel()  # should not raise
        assert group.started is False


class TestAsyncTaskGroupAstop:
    async def test_astop_group(self) -> None:
        t1 = SimpleTask()
        t2 = SimpleTask()
        group = AsyncTaskGroup([t1, t2])
        await group.astart()
        await group.ajoin()
        await group.astop()
        assert group.started is False
        assert t1.started is False
        assert t2.started is False

    async def test_astop_group_without_astart(self) -> None:
        group = AsyncTaskGroup([SimpleTask()])
        await group.astop()  # should not raise

    async def test_astop_running_group_raises(self) -> None:
        t1 = SlowTask(delay=10.0)
        group = AsyncTaskGroup([t1])
        await group.astart()
        with pytest.raises(ExceptionGroup):
            await group.astop()
        await group.acancel()
        await group.astop()


class TestAsyncTaskGroupAstart:
    async def test_astart_is_idempotent(self) -> None:
        t1 = SimpleTask()
        group = AsyncTaskGroup([t1])
        await group.astart()
        fut1 = group._fut
        await group.astart()
        fut2 = group._fut
        assert fut1 is fut2
        await group.ajoin()
        await group.astop()

    async def test_astart_starts_subtasks(self) -> None:
        t1 = SimpleTask()
        t2 = SimpleTask()
        group = AsyncTaskGroup([t1, t2])
        await group.astart()
        assert t1.started is True
        assert t2.started is True
        await group.ajoin()
        await group.astop()


class TestAsyncTaskGroupContextManager:
    async def test_async_with_group(self) -> None:
        t1 = SimpleTask()
        async with AsyncTaskGroup([t1]) as group:
            results = await group
        assert results == [22]
        assert group.started is False

    async def test_async_with_acancel_group(self) -> None:
        t1 = SlowTask(delay=10.0)
        async with AsyncTaskGroup([t1]) as group:
            await group.acancel()
        assert group.started is False


class TestAsyncTaskGroupFailure:
    async def test_subtask_failure_propagates(self) -> None:
        t1 = SimpleTask()
        t2 = FailingTask()
        group = AsyncTaskGroup([t1, t2])
        await group.astart()
        with pytest.raises(ExceptionGroup):
            await group.ajoin()
        await group.astop()
