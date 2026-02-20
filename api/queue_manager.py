import asyncio
import uuid
import time
from typing import Dict, Any, List, Optional, Callable
import logging
from pydantic import BaseModel

logger = logging.getLogger("qpyt-ui")

class TaskStatus(BaseModel):
    task_id: str
    status: str  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    type: str  # generate, upscale, etc.

class QueueManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QueueManager, cls).__new__(cls)
            cls._instance.queue = asyncio.Queue()
            cls._instance.tasks: Dict[str, TaskStatus] = {}
            cls._instance.worker_task = None
            cls._instance.is_running = False
        return cls._instance

    @classmethod
    def get_instance(cls):
        return cls()

    async def start_worker(self):
        if self.is_running:
            return
        self.is_running = True
        self.worker_task = asyncio.create_task(self._worker_loop())
        print("[QueueManager] Worker started.")

    async def stop_worker(self):
        self.is_running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        print("[QueueManager] Worker stopped.")

    async def add_task(self, task_type: str, func: Callable, *args, **kwargs) -> str:
        task_id = str(uuid.uuid4())
        task = TaskStatus(
            task_id=task_id,
            status="PENDING",
            created_at=time.time(),
            type=task_type
        )
        self.tasks[task_id] = task
        
        # We store a tuple of (task_id, func, args, kwargs)
        await self.queue.put((task_id, func, args, kwargs))
        print(f"[QueueManager] Task {task_id} added to queue ({task_type}).")
        return task_id

    async def _worker_loop(self):
        while self.is_running:
            try:
                task_id, func, args, kwargs = await self.queue.get()
                
                if task_id not in self.tasks or self.tasks[task_id].status == "CANCELLED":
                    self.queue.task_done()
                    continue

                task = self.tasks[task_id]
                task.status = "RUNNING"
                task.started_at = time.time()
                logger.info(f"[QueueManager] Processing task {task_id} ({task.type})...")

                try:
                    # Execute the actual generation function
                    # We expect func to be a callable that might be synchronous or asynchronous
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        # For blocking calls, we still run them in the main thread of the worker
                        # but since this IS the worker thread/loop, it's fine for serialized execution.
                        # Note: If we want progress updates, the func should ideally be async or 
                        # have a way to update the task object.
                        result = func(*args, **kwargs)
                    
                    task.result = result
                    task.status = "COMPLETED"
                    logger.info(f"[QueueManager] Task {task_id} completed.")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    logger.error(f"[QueueManager] Task {task_id} failed: {e}")
                    task.status = "FAILED"
                    task.error = str(e)
                finally:
                    task.completed_at = time.time()
                    task.progress = 100.0
                    self.queue.task_done()
                    
            except Exception as e:
                logger.error(f"[QueueManager] Worker loop error: {e}")
                await asyncio.sleep(1)

    def get_task(self, task_id: str) -> Optional[TaskStatus]:
        return self.tasks.get(task_id)

    def list_tasks(self) -> List[TaskStatus]:
        # Return last 20 tasks sorted by creation time
        return sorted(self.tasks.values(), key=lambda x: x.created_at, reverse=True)[:20]

    def cancel_task(self, task_id: str) -> bool:
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == "PENDING":
                task.status = "CANCELLED"
                return True
        return False

    def cancel_all(self):
        """Cancels all pending tasks and clears the queue."""
        logger.info("[QueueManager] Cancelling all pending tasks...")
        
        # 1. Mark all PENDING tasks as CANCELLED
        for task in self.tasks.values():
            if task.status == "PENDING":
                task.status = "CANCELLED"
        
        # 2. Empty the queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        logger.info("[QueueManager] All pending tasks cancelled and queue cleared.")

    def clear_completed(self):
        # Remove tasks that are completed, failed or cancelled to free memory
        to_remove = [tid for tid, t in self.tasks.items() if t.status in ["COMPLETED", "FAILED", "CANCELLED"]]
        for tid in to_remove:
            del self.tasks[tid]
