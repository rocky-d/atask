"""Lightweight async task abstraction over `asyncio`.

- [Homepage](https://pypi.org/project/atask/)
- [Repository](https://github.com/rocky-d/atask)
"""

import asyncio as aio
from contextvars import Context
from types import TracebackType
from typing import Any, Awaitable, Generator, Iterable, Self, Type, final

__all__ = [
    "AsyncTask",
    "AsyncTaskGroup",
]


class AsyncTask[T](Awaitable[T]):
    """Abstract base for an async task with lifecycle-managed resources.

    Bridges `asyncio.Task` with class resource management.

    Lifecycle:
        `__init__` -> `start` -> `_run` -> (`join` | `cancel`) -> `stop`

    Subclasses override `_run` to define task logic. Parameters are passed via
    `__init__`, not `_run`. Override `start`, `cancel`, `stop` for resource
    management (always call `super()`).

    Supports `await`, `async with`, and manual control flow.

    Examples:
        Define a task:

            class MyTask(AsyncTask[int]):
                async def _run(self) -> int:
                    await asyncio.sleep(1.0)
                    return 22

        Manual lifecycle:

            task = MyTask()
            await task.start()
            await task.join()
            task.result  # 22
            await task.stop()

        Async context manager:

            async with MyTask() as task:
                result = await task  # 22
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        context: Context | None = None,
    ) -> None:
        """Initializes the task in a finished, unstarted state.

        The internal future starts done with `asyncio.InvalidStateError` so
        `start` is the sole transition into running state.

        Args:
            name: Optional name forwarded to `asyncio.create_task`.
            context: Optional `contextvars.Context` for the task.
        """
        self._name = name
        self._context = context
        self._fut = aio.Future()
        self._fut.set_exception(aio.InvalidStateError())
        self._fut.exception()
        self._started = False

    @final
    def __await__(
        self,
    ) -> Generator[Any, None, T]:
        """Awaits task completion and returns the result."""
        yield from self.join().__await__()
        return self.result

    @final
    async def __aenter__(
        self,
    ) -> Self:
        """Starts the task for use as an async context manager."""
        await self.start()
        return self

    @final
    async def __aexit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        """Stops the task upon exiting the async context."""
        await self.stop(exc_type, exc_value, exc_traceback)

    @final
    @property
    def name(
        self,
    ) -> str | None:
        """The task name forwarded to `asyncio.create_task`."""
        return self._name

    @final
    @property
    def context(
        self,
    ) -> Context | None:
        """The `contextvars.Context` for the task."""
        return self._context

    @final
    @property
    def started(
        self,
    ) -> bool:
        """Whether resources have been acquired and not yet released."""
        return self._started

    @final
    @property
    def done(
        self,
    ) -> bool:
        """Whether the underlying future has completed."""
        return self._fut.done()

    @final
    @property
    def cancelled(
        self,
    ) -> bool:
        """Whether the task was cancelled."""
        return self._fut.cancelled()

    @final
    @property
    def running(
        self,
    ) -> bool:
        """Whether the task is currently executing (`started` and not `done`)."""
        return self.started and not self.done

    @final
    @property
    def result(
        self,
    ) -> T:
        """The task result. Raises if not done or if the task failed."""
        return self._fut.result()

    @final
    @property
    def exception(
        self,
    ) -> BaseException | None:
        """The exception raised by the task, or `None` on success."""
        return self._fut.exception()

    async def _run(
        self,
    ) -> T:
        """Defines the task's async logic.

        Subclasses must override this. Accepts no arguments beyond `self`;
        configure parameters via `__init__` instead.

        Returns:
            The task result of type `T`.

        Raises:
            NotImplementedError: If not overridden.
        """
        raise NotImplementedError()

    async def start(
        self,
    ) -> None:
        """Acquires resources and starts the task.

        No-op if already started. Overrides should acquire resources before
        calling `super().start()`.
        """
        if self.started:
            return
        self._started = True
        self._fut = aio.create_task(self._run(), name=self._name, context=self._context)

    @final
    async def join(
        self,
    ) -> None:
        """Waits for the task to complete. No-op if not started."""
        if not self.started:
            return
        await self._fut

    async def cancel(
        self,
        msg: Any | None = None,
    ) -> None:
        """Cancels the running task.

        No-op if not started. Overrides should cancel managed resources
        before calling `super().cancel()`.

        Args:
            msg: Optional cancellation message.
        """
        if not self.started:
            return
        self._fut.cancel(msg=msg)
        try:
            await self._fut
        except aio.CancelledError:
            pass

    async def stop(
        self,
        exc_type: Type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        exc_traceback: TracebackType | None = None,
    ) -> None:
        """Releases resources after the task has finished.

        Requires the task to be started and done. Raises if the task is
        still running. Overrides should release managed resources before
        calling `super().stop()`.

        Args:
            exc_type: Exception type from the async context manager, if any.
            exc_value: Exception value from the async context manager, if any.
            exc_traceback: Traceback from the async context manager, if any.

        Raises:
            asyncio.InvalidStateError: If the task is still running.
        """
        if not self.started:
            return
        if not self.done:
            raise aio.InvalidStateError() from exc_value
        self._started = False


class AsyncTaskGroup[T](AsyncTask[list[T]]):
    """An `AsyncTask` subclass that runs multiple tasks concurrently.

    Delegates lifecycle operations to all contained tasks. Results are
    collected as a list preserving insertion order.

    Examples:
        Run tasks concurrently:

            group = AsyncTaskGroup([TaskA(), TaskB()])
            async with group as g:
                results = await g  # [result_a, result_b]
    """

    def __init__(
        self,
        atsks: Iterable[AsyncTask[T]],
        *,
        name: str | None = None,
        context: Context | None = None,
    ) -> None:
        """Initializes the group with the given tasks.

        Args:
            atsks: `AsyncTask` instances to run concurrently.
            name: Optional group name forwarded to `asyncio.create_task`.
            context: Optional `contextvars.Context` for the group task.
        """
        super().__init__(name=name, context=context)
        self._atsks = list(atsks)

    async def _run(
        self,
    ) -> list[T]:
        """Joins all subtasks concurrently and collects their results.

        Returns:
            List of results in insertion order.
        """
        async with aio.TaskGroup() as tg:
            for atsk in self._atsks:
                tg.create_task(atsk.join())
        return [atsk.result for atsk in self._atsks]

    async def start(
        self,
    ) -> None:
        """Starts all subtasks concurrently, then starts the group task."""
        if self.started:
            return
        async with aio.TaskGroup() as tg:
            for atsk in self._atsks:
                tg.create_task(atsk.start())
        await super().start()

    async def cancel(
        self,
        msg: Any | None = None,
    ) -> None:
        """Cancels all subtasks concurrently, then cancels the group task.

        Args:
            msg: Optional cancellation message.
        """
        if not self.started:
            return
        async with aio.TaskGroup() as tg:
            for atsk in self._atsks:
                tg.create_task(atsk.cancel(msg))
        await super().cancel(msg)

    async def stop(
        self,
        exc_type: Type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        exc_traceback: TracebackType | None = None,
    ) -> None:
        """Stops all subtasks concurrently, then stops the group task.

        Args:
            exc_type: Exception type from the async context manager, if any.
            exc_value: Exception value from the async context manager, if any.
            exc_traceback: Traceback from the async context manager, if any.

        Raises:
            asyncio.InvalidStateError: If any task is still running.
        """
        if not self.started:
            return
        async with aio.TaskGroup() as tg:
            for atsk in self._atsks:
                tg.create_task(atsk.stop(exc_type, exc_value, exc_traceback))
        await super().stop(exc_type, exc_value, exc_traceback)
