import os

broker_url = os.getenv("NEUROGENPY_CELERY_BROKER", "redis://127.0.0.1:6379")
result_backend = os.getenv("NEUROGENPY_CELERY_RESULT",
                           "redis://127.0.0.1:6379")

worker_send_task_events = True
task_send_sent_event = True
