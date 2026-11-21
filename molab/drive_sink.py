class DriveSink:
    """Asynchronous durable telemetry; network latency never stalls a CUDA layer."""
    def __init__(self, config, directory):
        import threading
        self.config = config
        self.directory = directory
        self.pending = None
        self.lock = threading.Condition()
        self.ack = -1
        self.sequence = -1
        self.error = None
        self.received_utc = None
        self.last_report = None
        if config:
            import json
            outbox = directory / 'drive-outbox.json'
            if outbox.is_file():
                saved = json.loads(outbox.read_text(encoding='utf-8'))
                if saved.get('key') != config['key']:
                    raise ValueError('Drive outbox belongs to a different run')
                self.pending = saved
                self.sequence = saved['sequence']
            threading.Thread(target=self.worker, daemon=True).start()

    def snapshot(self):
        return dict(enabled=bool(self.config), sequence=self.sequence, saved_sequence=self.ack,
                    last_error=self.error, received_utc=self.received_utc)

    def enqueue(self, report):
        if not self.config:
            return
        import json, time
        with self.lock:
            from datetime import datetime, timezone
            snapshot = json.loads(json.dumps(report))
            snapshot['heartbeat'] = dict(utc=datetime.now(timezone.utc).isoformat(),
                message='Molab process is alive' if report.get('status') not in ('complete','partial','failed','gpu_unavailable') else 'Calculation finished',
                interval_seconds=60)
            self.last_report = snapshot
            self.sequence = max(self.sequence + 1, int(time.time() * 1000))
            self.pending = dict(op='put', key=self.config['key'], sequence=self.sequence,
                                report=snapshot, layers=report.get('layer_events', []))
            temporary = self.directory / 'drive-outbox.tmp'
            temporary.write_text(json.dumps(self.pending), encoding='utf-8')
            temporary.replace(self.directory / 'drive-outbox.json')
            self.lock.notify_all()

    def worker(self):
        import json, time, urllib.request
        delay = 2
        while True:
            with self.lock:
                while self.pending is None:
                    self.lock.wait(timeout=60)
                    if self.pending is None and self.last_report and self.last_report.get('status') not in ('complete','partial','failed','gpu_unavailable'):
                        from datetime import datetime, timezone
                        heartbeat = json.loads(json.dumps(self.last_report))
                        heartbeat['heartbeat'] = dict(utc=datetime.now(timezone.utc).isoformat(), message='Molab process is alive', interval_seconds=60)
                        self.enqueue(heartbeat)
                packet = self.pending.copy()
            try:
                payload = json.dumps(dict(token=self.config['token'], **packet)).encode('utf-8')
                request = urllib.request.Request(self.config['endpoint'], data=payload,
                    headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request, timeout=45) as response:
                    result = json.loads(response.read(1000000))
                if not result.get('ok'):
                    raise RuntimeError(str(result.get('error', 'Drive upload failed')))
                with self.lock:
                    self.ack = max(self.ack, packet['sequence'], result.get('sequence', -1))
                    self.received_utc = result.get('received_utc')
                    self.error = None
                    if self.pending and self.pending['sequence'] <= self.ack:
                        self.pending = None
                    self.lock.notify_all()
                delay = 2
                time.sleep(1)
            except Exception as error:
                self.error = f'{type(error).__name__}: {error}'
                time.sleep(delay)
                delay = min(delay * 2, 30)

    def flush(self, timeout=180):
        import time
        if not self.config:
            return True
        end = time.monotonic() + timeout
        with self.lock:
            while self.ack < self.sequence and time.monotonic() < end:
                self.lock.wait(timeout=min(5, max(0, end-time.monotonic())))
            return self.ack >= self.sequence
