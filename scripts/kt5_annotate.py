"""Ручная разметка ответов КТ 5: локальный веб-сервер со страницей разметки.

    python kt5_annotate.py

Выборку пишет ноутбук `kt5_generation` (`gold/kt5_annotation_sample.json`). Каждая сохранённая оценка сразу
записывается в `gold/kt5_human_labels.json`, поэтому разметку можно прервать и продолжить.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

from gigarag import annotation
from settings import ANNOTATION_HOST, ANNOTATION_PORT, ANNOTATION_SAMPLE, HUMAN_LABELS

records = annotation.read_sample(ANNOTATION_SAMPLE)
known_ids = {r["id"] for r in records}
page = annotation.page(records).encode()
write_lock = Lock()


class Handler(BaseHTTPRequestHandler):
    def reply(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def reply_labels(self) -> None:
        self.reply(200, json.dumps(annotation.read_labels(HUMAN_LABELS), ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if self.path == "/":  # продолжить с первого неразмеченного ответа
            self.send_response(302)
            first = annotation.first_unlabeled(records, annotation.read_labels(HUMAN_LABELS))
            self.send_header("Location", f"/{first}")
            self.end_headers()
        elif self.path.strip("/").isdigit() and 1 <= int(self.path.strip("/")) <= len(records):
            self.reply(200, page, "text/html; charset=utf-8")
        elif self.path == "/labels":
            self.reply_labels()
        else:
            self.reply(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/labels":
            return self.reply(404, b"not found", "text/plain")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if body.get("id") not in known_ids:
            return self.reply(400, b"unknown id", "text/plain")
        with write_lock:
            annotation.write_labels(HUMAN_LABELS, annotation.with_label(
                annotation.read_labels(HUMAN_LABELS), body["id"], body.get("scores", {})))
        self.reply_labels()


def main() -> None:
    done = sum(annotation.is_complete(annotation.read_labels(HUMAN_LABELS), r["id"]) for r in records)
    print(f"{len(records)} ответов, размечено {done}; http://{ANNOTATION_HOST}:{ANNOTATION_PORT}/")
    ThreadingHTTPServer((ANNOTATION_HOST, ANNOTATION_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
