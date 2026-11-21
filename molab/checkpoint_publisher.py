class SnapshotPublisher:
    """Transfer complete, immutable layer snapshots in one background thread."""
    def __init__(self, transport, root, identity, status):
        from concurrent.futures import ThreadPoolExecutor
        self.transport = transport
        self.root = root
        self.identity = identity
        self.status = status
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.active = None
        self.committed = -1
        self.retry_at = 0
        self.scan_at = 0

    def poll(self, final=False):
        import time
        import shutil
        now = time.monotonic()
        if self.pending and (final or self.pending.done()):
            try:
                receipt = self.pending.result()
            except Exception as error:
                self.status.update(state='upload_failed', error=str(error)[:1000])
                self.retry_at = now + 60
            else:
                self.committed = self.active[0]
                self.status.update(state='committed', next_depth=self.committed, receipt=receipt)
                self.status.pop('error', None)
                for layer, path in self.ready():
                    if layer <= self.committed:
                        shutil.rmtree(path)
            self.pending = self.active = None
        if self.pending or now < self.retry_at or (now < self.scan_at and not final):
            return
        self.scan_at = now + 5
        choices = self.ready()
        if choices:
            self.active = max(choices)
            layer, folder = self.active
            self.status.update(state='uploading', uploading_depth=layer)
            self.pending = self.pool.submit(self.transport.upload, folder, self.identity, layer)

    def ready(self):
        import re
        return [(int(m[1]), path) for path in self.root.glob('next-*.ready')
                if (m := re.fullmatch(r'next-([0-9]+)\.ready', path.name)) and path.is_dir()]

    def close(self):
        try:
            self.poll(final=True)
            if self.pending:
                self.poll(final=True)
        finally:
            self.pool.shutdown(wait=True)
