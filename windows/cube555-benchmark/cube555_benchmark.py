"""Autonomous Molab experiments, exclusively through Cube555-Control buttons.

There are no network clients, model calls, or solver implementations here.
Every cloud action goes through a serialized, durable request to form 1.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import uuid

ROOT = Path(sys.executable if getattr(sys, 'frozen', False) else __file__).resolve().parent
DATA = ROOT / 'runs'
FIRST = ROOT.parent / 'cube555-control' / 'Cube555-Control.exe'
POLL_SECONDS = 300
CURRENT_CSV = Path.home() / 'Downloads/submission_555_100284.csv'
MAX_RUNNING = 1
LIST_RETRY_SECONDS = 60
SESSION_LIMIT = 5 * 3600
CLEANUP_MARGIN = 15 * 60
FIRST_WRITE_TIMEOUT = 20 * 60
RETRYABLE = {'cloud_unavailable', 'gpu_unavailable', 'not_started', 'import_absent_from_inventory',
             'imported_no_session', 'cloud_absent_without_terminal_report', 'no_first_drive_write'}
MOSCOW = timezone(timedelta(hours=3))
END_STATUSES = {'complete', 'partial', 'failed', 'gpu_unavailable', 'stopped'}
PLANNED_MODES = {'diagnostics', 'repair', 'tpu_controls'}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError, TypeError):
        return default


def load_lengths(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        lengths = {}
        for row in csv.DictReader(source):
            cube = int(row['initial_state_id'])
            moves = row['path'].split('.') if row['path'] else []
            if cube in lengths or any(not re.fullmatch(r'-?[rdf][0-4]', move) for move in moves):
                raise ValueError(f'Некорректная строка кубика {cube} в {path.name}')
            lengths[cube] = len(moves)
    if set(lengths) != set(range(1035)):
        raise ValueError(f'{path.name} должен содержать 1035 разных кубиков.')
    return lengths


def baseline_lengths():
    candidates = [CURRENT_CSV, ROOT.parent / 'user-solutions/dataset/shortest_paths_result_1.csv',
                  Path.home() / 'Downloads/shortest_paths_result (1).csv']
    for path in candidates:
        if path.is_file():
            return load_lengths(path)
    return {}


def night_deadline(now):
    local = datetime.fromtimestamp(now, MOSCOW)
    noon = local.replace(hour=12, minute=0, second=0, microsecond=0)
    if noon <= local:
        noon += timedelta(days=1)
    return min(now + 10 * 3600, noon.timestamp())


def historical_layer_rate(directory):
    """Median of per-job medians, in seconds per 1M actual retained states."""
    medians = []
    for path in directory.glob('*/job.json'):
        job = read_json(path, {})
        layers = job.get('depth_timings') or (job.get('current_case') or {}).get('depth_timings') or []
        rates = [r['seconds'] * 1e6 / r['frontier_size'] for r in layers if r.get('saturated') and
                 r.get('depth',0) >= 7 and r.get('frontier_size',0) > 0 and r.get('seconds',0) > 0]
        if rates:
            medians.append(statistics.median(rates))
    return statistics.median(medians) if medians else 17.7


def diagnostic_plan(lengths, seed):
    rng = random.Random(seed)
    frames = list(range(1, 48))
    rng.shuffle(frames)
    cases = []
    def pair(cube, beam, depth, symmetry, purpose, trace=False):
        directions = [False,True]
        if cases:
            rng.shuffle(directions)
        for inv in directions:
            cases.append(dict(cube=cube, beam=beam, requested_depth=depth, symmetry=symmetry,
                              inverse=inv, purpose=purpose, trace_reference=trace,
                              lane='diagnostic', case_number=len(cases)+1))
    pair(45, 200_000, 18, 0, 'Контроль прямого и обратного поиска')
    pair(68, 1_100_000, 45, 23, 'Повтор случая 68/S23 и наблюдение ветки известного пути', True)
    pair(45, 3_500_000, 18, frames[0], 'Короткий кубик на целевом луче в случайной симметрии')
    pair(68, 3_500_000, 45, 23, 'Эффект ширины при той же симметрии', True)
    pair(68, 3_500_000, 45, 0, 'Контроль без смены системы координат', True)
    eligible = [k for k,v in lengths.items() if v>=30 and k!=68]
    if not eligible:
        raise ValueError('В CSV нет независимого кубика с путём от 30 ходов.')
    delta = min(abs(lengths[k]-36) for k in eligible)
    cube36 = rng.choice(sorted(k for k in eligible if abs(lengths[k]-36)==delta))
    pair(cube36, 3_500_000, lengths[cube36]+10, frames[0],
         f'Независимый кубик с исходной длиной {lengths[cube36]}')
    random_68_frame = next(f for f in frames if f != 23)
    pair(68, 3_500_000, 45, random_68_frame, 'Случайная симметрия кубика 68', True)
    for length in [32,33,36,41]:
        ids=sorted(k for k,v in lengths.items() if v==length and k not in (68,cube36))
        if ids:
            pair(rng.choice(ids),3_500_000,length+10,rng.choice(frames),'Поиск улучшения исходного пути')
    for frame in frames[1:]:
        pair(45, 200_000, 18, frame, 'Резерв свободного времени: короткая пара')
    return cases


def tpu_control_plan(lengths):
    # Cube 994 uses the exact frame of artgor TPU v56's verified 110-move solution.
    examples = [(76, 40, 0, False), (142, 100, 0, False), (994, 110, 45, True),
                (82, 40, 12, True), (166, 100, 23, False), (170, 110, 31, True)]
    cases = []
    for cycle in range(48):
        for cube, length, frame, inverse in examples:
            if lengths.get(cube) != length:
                raise ValueError(f'Кубик {cube}: ожидалось {length} ходов в исходном CSV.')
            cases.append(dict(cube=cube, beam=4_194_304, requested_depth=length + 12,
                              symmetry=(frame+cycle) % 48, inverse=inverse ^ bool(cycle % 2),
                              case_number=len(cases)+1, lane='tpu_controls',
                              purpose=f'Контроль поиска Cube555 для пути длиной {length}'))
    return cases


def unwrap_report(value):
    if isinstance(value, dict) and isinstance(value.get('result'), dict):
        return value['result']
    return value if isinstance(value, dict) else {}


def verified_path(test):
    for solution in (test.get('verified_solution'), (test.get('job') or {}).get('solution')):
        if not isinstance(solution, dict) or solution.get('verified') is not True:
            continue
        path, length = solution.get('path'), solution.get('length')
        if isinstance(path, str) and type(length) is int:
            moves = path.split('.') if path else []
            if len(moves) == length and all(re.fullmatch(r'-?[rdf][0-4]', move) for move in moves):
                return solution
    return {}


def met_target(test, baseline):
    solution = verified_path(test)
    return bool(solution) and type(baseline) is int and solution['length'] <= baseline


def repair_plan(previous, lengths):
    """Keep the original seeded cases, including unstarted cases after an interrupted repair."""
    attempts = {}
    for test in previous['jobs']:
        attempts.setdefault(test['case_number'], []).append(test)
    remaining = []
    for case in previous['plan']:
        history = attempts.get(case['case_number'], [])
        if any(met_target(t, lengths[case['cube']]) for t in history):
            continue
        last = history[-1] if history else {}
        remaining.append(dict(case, previous_outcome=last.get('outcome', 'not_started'),
                              previous_test_id=last.get('test_id'), repair_stage=2))
    if remaining:
        small = [c for c in remaining if c['previous_outcome'] == 'cuda_illegal_access']
        first = min(small or remaining, key=lambda c: (c['beam'], c['requested_depth'], c['case_number']))
        first['repair_stage'] = 0
        if small:
            first['cuda_diagnostics'] = 'sync'
        hard = sorted((c for c in remaining if c['cube'] == 68 and c is not first),
                      key=lambda c: (c['beam'], c['case_number']))
        if hard:
            pair = hard[0]
            for case in hard:
                if (case['beam'], case['symmetry'], case['requested_depth']) == (
                        pair['beam'], pair['symmetry'], pair['requested_depth']):
                    case['repair_stage'] = 1
            pair['cuda_diagnostics'] = 'memcheck'
    return sorted(remaining, key=lambda c: (c['repair_stage'], c['case_number']))


def classify(job, report=None):
    """Never interpret an HTTP/session failure as evidence of a beam limit."""
    report = report or {}
    attempts = report.get('attempts') or []
    text = '\n'.join(str(s or '') for s in [job.get('error'), report.get('error'), report.get('last_log'),
                     job.get('last_log'), *(r.get('diagnostic') for r in attempts)]).lower()
    if verified_path({'job':job}):
        return 'solution'
    if 'illegal memory access' in text or 'illegal address' in text:
        return 'cuda_illegal_access'
    if any(s in text for s in ('cuda out of memory', 'cuda_error_out_of_memory',
                              'std::bad_alloc', 'out of memory', 'cudamalloc failed')):
        return 'resource_limit'
    if re.search(r'(?:http(?: error)?|status(?: code)?)[ :=]+(?:403|404|502|503)\b', text) or any(
            s in text for s in ('session expired', 'disconnected', 'connection reset', 'cloud_terminated', 'runtime unavailable')):
        return 'cloud_unavailable'
    if report.get('status') == 'gpu_unavailable' or job.get('status') == 'gpu_unavailable':
        return 'gpu_unavailable'
    reasons = [report.get('stop_reason'), *(r.get('stop_reason') for r in attempts),
               *(r.get('search_outcome') for r in attempts)]
    if any(r in ('frontier_exhausted','empty_frontier','beam_exhausted') for r in reasons):
        return 'frontier_exhausted'
    if any(r in ('depth_reached_without_solution', 'depth_reached', 'depth_limit') for r in reasons):
        return 'depth_reached'
    if job.get('status') in END_STATUSES:
        if job.get('status') in {'failed', 'partial'}:
            return 'native_error'
        return 'no_verified_path_reason_unknown' if job.get('status') == 'complete' else 'stopped'
    if job.get('error'):
        return 'unclassified_error'
    return 'running'


def layer_samples(job, report):
    """Return measured completed layers >=7, retaining actual (not requested) beam.

    New reports contain rank/depth/frontier_size/seconds. Older timing records
    without actual frontier_size are excluded rather than fabricating a beam.
    Native rank-log tails provide a conservative fallback for older reports.
    """
    found = {}
    attempts = list(report.get('attempts', []))
    if (job.get('current_case') or {}).get('depth_timings'):
        attempts.append(dict(depth_timings=job['current_case']['depth_timings'],
                             effective_beam=job['current_case'].get('effective_beam')))
    for attempt in attempts:
        effective = attempt.get('effective_beam') or job.get('effective_beam')
        for row in attempt.get('depth_timings', []) + attempt.get('layers', []):
            if not isinstance(row, dict):
                continue
            depth = row.get('depth', row.get('depth_index'))
            frontier = row.get('frontier_size', row.get('frontier'))
            seconds = row.get('seconds', row.get('depth_sec'))
            if depth is None or frontier is None or seconds is None or int(depth) < 7:
                continue
            rank = row.get('rank', 0)
            found[(str(rank), int(depth))] = dict(rank=rank, depth=int(depth),
                frontier_size=int(frontier), seconds=float(seconds), effective_beam=effective,
                started_utc=row.get('started_utc'), ended_utc=row.get('ended_utc'),
                full_beam=bool(effective and int(frontier) == int(effective) and row.get('saturated', True)))
    # Tails may have repeated native stdout plus rank log sections. Collapse by rank/depth.
    log = job.get('last_log', '')
    starts = {}
    rank = '0'
    for line in log.splitlines():
        section = re.search(r'rank-(\d+)\.log', line)
        if section:
            rank = section[1]
        prefix = re.search(r'\[default(\d+)\]', line)
        current_rank = prefix[1] if prefix else rank
        start = re.search(r'depth_start=(\d+) frontier_size=(\d+)', line)
        if start:
            starts[(current_rank, int(start[1]))] = int(start[2])
        done = re.search(r'depth_done=(\d+) depth_sec=([\d.eE+-]+)', line)
        if done:
            key = (current_rank, int(done[1]))
            if key in starts and key[1] >= 7 and key not in found:
                effective = job.get('effective_beam') or (job.get('solution') or {}).get('effective_beam')
                found[key] = dict(rank=current_rank, depth=key[1], frontier_size=starts[key],
                    seconds=float(done[2]), effective_beam=effective,
                    full_beam=bool(effective and starts[key] == int(effective)))
    return list(found.values())


def completed_layers(job, report):
    """Depth indices start at zero. Completing depth d establishes d+1 completed layers."""
    counts = []
    attempts = [a for a in report.get('attempts') or [] if a.get('kind','solve') == 'solve'
                and (a.get('puzzle_id') is None or job.get('cube') in (None,a['puzzle_id']))]
    for source in (job, report, job.get('current_case') or {}, report.get('current_case') or {}, *attempts):
        if type(source.get('completed_layers')) is int:
            counts.append(source['completed_layers'])
        for row in (source.get('depth_timings') or []) + (source.get('layer_events') or []):
            depth = row.get('depth')
            if type(depth) is not int or depth < 0 or row.get('puzzle_id',job.get('cube')) != job.get('cube'):
                continue
            if row.get('ended_utc') or row.get('seconds') is not None:
                counts.append(depth+1)
            elif row.get('started_utc'):
                counts.append(depth)
        log = source.get('last_log') or ''
        counts.extend(int(match[1])+1 for match in re.finditer(r'\bdepth_done=(\d+)',log))
        counts.extend(int(match[1])+1 for match in re.finditer(r'\bsolve_bucket_stop=1\s+puzzle_id=\d+\s+depth_index=(\d+)',log))
        if re.search(r'\bdepth_start=0\b', log):
            counts.append(0)
    return max(counts, default=None)


class Controller:
    def __init__(self, first=FIRST, data=DATA, clock=time.time, invoke=None):
        self.first, self.data, self.clock = Path(first), Path(data), clock
        self.invoke = invoke or self.invoke_exe
        self.data.mkdir(parents=True, exist_ok=True)
        self.pointer = self.data / 'current.json'
        pointer = read_json(self.pointer, {})
        self.directory = Path(pointer['directory']) if pointer.get('directory') else None
        self.state = read_json(self.directory / 'state.json', {}) if self.directory else {}
        if self.state.get('status') == 'stopping' and self.state.get('message','').startswith('Сохраняем результаты'):
            self.state.setdefault('user_stopped', True)
        self.commands = queue.Queue()
        self.snapshot = {}
        self.lengths = baseline_lengths()
        if (self.state.get('status') == 'running' and self.state.get('listing') is None
                and self.state.get('message','').startswith('Список Molab недоступен')
                and 'last_list_failure' not in self.state):
            last_failure = self.state.get('next_list', self.clock()) - POLL_SECONDS
            self.state.update(last_list_failure=last_failure, next_list=last_failure + LIST_RETRY_SECONDS)
            self.persist()
        if self.state and self.state.get('layer_count_version',0) < 1:
            for test in self.state.get('jobs',[]):
                job = test.get('job') or {}
                test['completed_layers'] = completed_layers(job, self.report_for(job))
            self.state['layer_count_version'] = 1
            self.persist()
            self.export()
        self.refresh_snapshot()

    @property
    def request_directory(self):
        return self.first.parent / 'runs/form-requests'

    def persist(self):
        if self.directory:
            atomic_json(self.directory / 'state.json', self.state)
        self.refresh_snapshot()

    def event(self, kind, **details):
        if self.directory:
            with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as output:
                output.write(json.dumps(dict(time=self.clock(), event=kind, **details), ensure_ascii=False) + '\n')

    def refresh_snapshot(self):
        self.snapshot = json.loads(json.dumps(self.state)) if self.state else {}

    def reconcile_local_launches(self):
        """Recover a form card; only proven pre-import failures release a slot."""
        if not self.state or self.state.get('pending'):
            return
        changed = False
        for test in self.state.get('jobs', []):
            if test.get('state') != 'uncertain_launch':
                continue
            job = test.get('job') or {}
            if job.get('key'):
                card_dir = self.first.parent / 'runs' / job['key']
                saved = read_json(card_dir / 'job.json', {})
                if saved.get('request_id') == test.get('launch_request_id') and saved.get('key') == job['key']:
                    job = saved
                    test['job'] = job
                diagnostic = read_json(card_dir / 'error.json', {})
            else:
                diagnostic = {}
            if job.get('id'):
                test.update(state='running', next_poll=self.clock())
                self.event('launch_reconciled', test_id=test['test_id'], notebook_id=job['id'])
                changed = True
            elif job.get('status') == 'error' and diagnostic.get('stage') == 'preparing':
                test.update(state='finished', outcome='not_started', released=True, finished=self.clock())
                self.event('preimport_failure_confirmed', test_id=test['test_id'], error=job.get('error'))
                changed = True
        if changed:
            if self.state.get('jobs') and all(t.get('state') == 'finished' for t in self.state['jobs']):
                if self.state.get('status') == 'stopping' or (
                        self.state.get('mode') == 'single' and self.state.get('status') == 'running'):
                    self.state.update(status='finished', finished=self.clock(), message='Все запуски завершены; отказ до импорта сохранён.')
            self.export()
            self.persist()

    def start(self, symmetry=3, hours=24, single=None, diagnostics=False, seed=None,
              baseline=CURRENT_CSV, repair=None, tpu_controls=False):
        self.reconcile_local_launches()
        if self.state.get('status') in {'running', 'paused', 'stopping','resuming'}:
            raise ValueError('Текущая серия ещё активна.')
        if type(symmetry) is not int or not 0 <= symmetry < 48:
            raise ValueError('Симметрия должна быть целым числом 0–47.')
        if single:
            cube, beam, depth = (single[k] for k in ('cube', 'beam', 'depth'))
            if type(cube) is not int or not 0 <= cube <= 1034:
                raise ValueError('Номер кубика должен быть целым числом 0–1034.')
            if type(beam) is not int or not 1 <= beam < 2**64:
                raise ValueError('Луч должен быть положительным целым uint64.')
            if type(depth) is not int or not 1 <= depth < 2**32:
                raise ValueError('Глубина должна быть положительным целым uint32.')
        if not self.first.is_file():
            raise FileNotFoundError(f'Не найдена первая форма: {self.first}')
        plan = None
        if diagnostics or repair or tpu_controls:
            if repair:
                state_file = Path(repair).resolve()
                if state_file.is_dir():
                    state_file /= 'state.json'
                previous = read_json(state_file, {})
                if previous.get('mode') not in {'diagnostics','repair'} or previous.get('status') != 'finished':
                    raise ValueError('Выберите state.json завершённой серии диагностики или повтора.')
                source = state_file.parent / 'baseline.csv'
                seed = previous['seed']
            else:
                hours = 10
                seed = random.SystemRandom().getrandbits(32) if seed is None else int(seed)
                source = Path(baseline)
            baseline_data = source.read_bytes()
            self.lengths = load_lengths(source)
            if repair and hashlib.sha256(baseline_data).hexdigest() != previous['baseline_sha256']:
                raise ValueError('Контрольная сумма baseline.csv отличается от исходного эксперимента.')
            if tpu_controls:
                plan = tpu_control_plan(self.lengths)
            else:
                plan = repair_plan(previous, self.lengths) if repair else diagnostic_plan(self.lengths, seed)
        now = self.clock()
        deadline = night_deadline(now) if diagnostics else now + hours * 3600
        self.directory = self.data / (time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
        self.directory.mkdir(parents=True)
        self.state = dict(version=4, layer_count_version=1, mode='tpu_controls' if tpu_controls else 'repair' if repair else 'diagnostics' if diagnostics else 'single' if single else 'benchmark', single=single,
            status='running', started=now, deadline=deadline,
            symmetry=int(symmetry), poll_seconds=POLL_SECONDS, jobs=[], pending=None,
            listing=None, listed_at=0, next_list=0, message='Проверка работающих ноутбуков через форму 1',
            baseline_lengths={str(k): self.lengths.get(k) for k in (57, 67, 94, 124, 145)})
        if diagnostics or repair or tpu_controls:
            rate = historical_layer_rate(self.first.parent / 'runs')
            # Both directions run in the same slot, one after the other.
            budget = deadline - now - CLEANUP_MARGIN
            planned = []
            for a,b in zip(plan[::2], plan[1::2]):
                estimated = math.ceil(600 + rate * a['beam']/1e6 * (a['requested_depth']-3) * 1.18)
                if 2 * estimated > budget or estimated > SESSION_LIMIT-CLEANUP_MARGIN:
                    continue
                planned.extend([dict(a, estimated_seconds=estimated),dict(b, estimated_seconds=estimated)])
                budget -= 2 * estimated
            if tpu_controls:
                for case in plan:
                    case['estimated_seconds'] = math.ceil(600 + rate * case['beam']/1e6 * (case['requested_depth']-3))
            elif not repair:
                plan = planned
            else:
                self.state['repair_source'] = str(state_file)
            (self.directory / 'baseline.csv').write_bytes(baseline_data)
            self.state.update(plan=plan, seed=seed, baseline_source=str(source), baseline_score=sum(self.lengths.values()),
                              session_limit_seconds=SESSION_LIMIT, shutdown_margin_seconds=CLEANUP_MARGIN,
                              estimated_seconds_per_million=rate, reference='comparison-only; native tracing only for cube 68',
                              baseline_sha256=hashlib.sha256(baseline_data).hexdigest(),
                              baseline_lengths={str(k): self.lengths[k] for k in {c['cube'] for c in plan}})
        atomic_json(self.pointer, {'directory': str(self.directory)})
        self.event('started', hours=hours, symmetry=symmetry, mode=self.state['mode'], single=single)
        self.persist()

    def invoke_exe(self, arguments):
        environment = dict(os.environ, PYINSTALLER_RESET_ENVIRONMENT='1')
        subprocess.Popen([str(self.first), *arguments], cwd=str(self.first.parent), env=environment,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def request(self, action, test=None, target_id=None):
        assert not self.state.get('pending'), 'Form 1 requests must be serialized'
        request_id = uuid.uuid4().hex
        arguments = [f'--{action}', '--request-id', request_id]
        if action == 'launch':
            arguments += ['--managed', '--cube', str(test['cube']), '--beam', str(test['beam']),
                          '--depth', str(test.get('requested_depth', 120)), '--symmetry', str(test.get('symmetry', self.state['symmetry']))]
            if test.get('inverse',False):
                arguments.append('--inverse')
            if test.get('trace_reference',False):
                arguments.append('--trace-reference')
            if test.get('cuda_diagnostics'):
                arguments.extend(['--cuda-diagnostics',test['cuda_diagnostics']])
            test['launch_request_id'] = request_id
        elif action in {'result', 'shutdown'}:
            arguments += ['--job', target_id or test['job']['key']]
        pending = dict(id=request_id, action=action, test_id=test['test_id'] if test else None,
                       target_id=target_id, arguments=arguments, created=self.clock(), dispatched=False)
        self.state['pending'] = pending
        self.persist()  # durable intent comes before starting another process
        self.dispatch_pending()

    def dispatch_pending(self):
        pending = self.state['pending']
        if pending['dispatched']:
            return
        try:
            self.invoke(pending['arguments'])
        except OSError as exc:
            pending.update(dispatched=False, retry_reason=f'Не удалось открыть форму 1: {exc}',
                           next_retry=self.clock()+LIST_RETRY_SECONDS)
            self.persist()
            return
        pending.update(dispatched=True, next_retry=self.clock()+LIST_RETRY_SECONDS)
        self.event('form_request', **pending)
        self.persist()

    def lookup(self, test_id):
        return next(t for t in self.state['jobs'] if t['test_id'] == test_id)

    def report_for(self, job):
        path = job.get('drive_report_file') or job.get('result_file')
        return unwrap_report(read_json(path, {})) if path else {}

    def consume(self):
        pending = self.state.get('pending')
        if not pending:
            return False
        receipt_path = self.request_directory / (pending['id'] + '.receipt')
        receipt = read_json(receipt_path)
        now = self.clock()
        test = self.lookup(pending['test_id']) if pending.get('test_id') else None
        cleanup = self.state.get('resume_cleanup')
        if receipt and (receipt.get('code') == 'screen_busy' or receipt.get('retryable')):
            if test is not None and isinstance(receipt.get('job'),dict):
                test['job'] = {**(test.get('job') or {}), **receipt['job']}
            wait = receipt.get('retry_at')
            wait = wait if isinstance(wait,(int,float)) else now + LIST_RETRY_SECONDS
            if 'retry_at' not in receipt and pending.get('last_retry_receipt') == receipt_path.stat().st_mtime_ns:
                wait = pending.get('next_retry',wait)
            pending.update(dispatched=False, deferred=True, next_retry=wait, last_retry_receipt=receipt_path.stat().st_mtime_ns,
                           retry_reason=str(receipt.get('error') or 'Экран занят другим приложением'))
            self.state['message'] = pending['retry_reason']
            self.persist()
            receipt_path.replace(receipt_path.with_suffix('.last_retry'))
            receipt = None
        if receipt is None:
            if pending.get('deferred') and pending['action'] == 'launch' and self.state['status'] != 'running':
                if self.state['status'] == 'paused':
                    self.state['message'] = 'Пауза. Отложенное нажатие ▶ приостановлено.'
                    self.refresh_snapshot()
                    return True
                if test is not None:
                    test.update(state='uncertain_launch')
                self.state['pending'] = None
                self.persist()
                return True
            if pending.get('deferred') and pending.get('target_id') and self.state['status'] != 'resuming':
                self.state['pending'] = None
                self.persist()
                return True
            due = pending.get('next_retry', pending['created'] + (LIST_RETRY_SECONDS if pending.get('dispatched') else 0))
            if now >= due:
                # A timeout cannot prove whether Upload/Play ran. Reopen the IPC
                # worker with the same idempotency key and its persisted Form 1 card.
                pending.update(dispatched=False, deferred=False)
                pending.pop('retry_reason',None)
                self.dispatch_pending()
            age = now - pending['created']
            self.state['message'] = pending.get('retry_reason') or f"Форма 1: {pending['action']}, ожидание ответа {int(age)} с"
            self.refresh_snapshot()
            return True
        action = pending['action']
        self.event('form_receipt', request_id=pending['id'], action=action, receipt=receipt)
        if action == 'list':
            listing = receipt.get('listing')
            if (receipt.get('error') or not isinstance(listing, dict)
                    or not isinstance(listing.get('running'), list) or listing.get('warnings')):
                detail = receipt.get('error')
                if not detail and isinstance(listing, dict):
                    detail = listing.get('warnings')
                now = self.clock()
                self.state.update(listing=None, listed_at=0, last_list_failure=now, next_list=now + LIST_RETRY_SECONDS,
                                  message='Список Molab недоступен. Повтор через 60 с. ' + str(detail or 'нет полного списка running'))
                if cleanup:
                    cleanup.update(inventory_valid=False, next_check=now+LIST_RETRY_SECONDS)
            else:
                self.state.update(listing=listing, listed_at=self.clock(), next_list=self.clock() + POLL_SECONDS)
                # A missing heartbeat alone does not establish that a GPU session has stopped.
                running_ids = {r['id'] for r in listing['running'] if isinstance(r, dict) and r.get('id')}
                if cleanup:
                    if len(running_ids) != len(listing['running']):
                        cleanup.update(inventory_valid=False, next_check=now+LIST_RETRY_SECONDS)
                        self.state['message'] = 'Molab не сообщил ID всех сессий. Повтор проверки через минуту.'
                    else:
                        cleanup.update(inventory_valid=True, running_ids=sorted(running_ids), next_check=now)
                elif self.state.get('mode') in PLANNED_MODES or self.state.get('status') == 'stopping':
                    for entry in self.state['jobs']:
                        if entry.get('state') == 'uncertain_launch':
                            job = entry.get('job') or {}
                            # A confirmed empty running inventory releases the slot on explicit Stop.
                            # Preserve the ambiguous import outcome for the audit.
                            if self.state.get('user_stopped') and job.get('title') and not listing['running']:
                                entry.update(state='finished', outcome='stopped_no_running_session',
                                             released=True, finished=self.clock())
                                continue
                            # Only a complete inventory with a unique, known form title
                            # can resolve an ambiguous import. Do not take over a live ID.
                            notebooks = listing.get('notebooks')
                            if job.get('title') and isinstance(notebooks, list) and not listing.get('warnings'):
                                matches = [n for n in notebooks if n.get('name') == job['title']]
                                if len(matches)==1 and matches[0].get('id') not in running_ids:
                                    entry.update(state='finished',outcome='imported_no_session',released=True,finished=self.clock())
                                elif len(matches)==1 and matches[0].get('id') in running_ids:
                                    entry['job'].update(id=matches[0]['id'], url=matches[0].get('url'))
                                    entry.update(state='running', next_poll=self.clock())
                                    self.event('unique_title_reconciled', test_id=entry['test_id'], notebook_id=matches[0]['id'])
                                elif matches:
                                    entry['absent_title_at'] = None
                                elif entry.get('absent_title_at') is None:
                                    entry['absent_title_at'] = self.clock()
                                elif self.clock() - entry['absent_title_at'] >= POLL_SECONDS:
                                    entry.update(state='finished', outcome='import_absent_from_inventory', released=True,
                                                 finished=self.clock())
                            continue
                        if (entry.get('state') != 'running' or not entry.get('check_cloud_presence')
                                or listing.get('warnings')):
                            continue
                        entry['check_cloud_presence'] = False
                        if entry['job']['id'] in running_ids:
                            entry['missing_confirmations'] = 0
                        else:
                            entry['missing_confirmations'] = entry.get('missing_confirmations', 0) + 1
                            if entry['missing_confirmations'] >= 2:
                                entry.update(state='finished', outcome='cloud_absent_without_terminal_report',
                                             finished=self.clock(), released=True)
        elif pending.get('target_id'):
            if cleanup:
                target = pending['target_id']
                result = receipt.get('job') or {}
                stopped = (result.get('id') == target and result.get('session_released') and not receipt.get('error'))
                if stopped:
                    if target not in cleanup['closed_ids']:
                        cleanup['closed_ids'].append(target)
                else:
                    self.state['message'] = f"Завершение {target}: {receipt.get('error') or 'нет подтверждения'}"
                cleanup.update(inventory_valid=False, next_check=now + (0 if stopped else LIST_RETRY_SECONDS))
        elif test is not None:
            old_job = test.get('job') or {}
            job = {**old_job, **(receipt.get('job') or {})}
            test['job'] = job
            test['last_poll'] = self.clock()
            test['next_poll'] = self.clock() + POLL_SECONDS
            if receipt.get('error'):
                test['last_error'] = str(receipt['error'])
            else:
                test.pop('last_error', None)
            report = self.report_for(job)
            count = completed_layers(job, report)
            if count is not None:
                test['completed_layers'] = max(test.get('completed_layers') or 0, count)
            samples = layer_samples(job, report)
            old = {(str(x['rank']), x['depth']): x for x in test.get('layers', [])}
            old.update({(str(x['rank']), x['depth']): x for x in samples})
            test['layers'] = list(old.values())
            if action != 'shutdown':
                test['classification'] = classify(job, report)
            solution = verified_path(test)
            if solution and isinstance(test.get('baseline_length'),int):
                test['verified_solution'] = solution.copy()
                test['improvement'] = test['baseline_length'] - solution['length']
            if action == 'shutdown':
                stopped = (job.get('session_released') or job.get('status') == 'stopped') and not receipt.get('error')
                test['released'] = stopped
                test['next_shutdown'] = self.clock() + LIST_RETRY_SECONDS
                if stopped:
                    test['finished'] = self.clock()
                    test['state'] = 'finished'
                    self.state['listed_at'] = 0
                else:
                    test['state'] = 'releasing'
            elif not job.get('id'):
                # No automatic relaunch after an uncertain import/start response.
                test.update(state='uncertain_launch', last_error=receipt.get('error') or 'Нет ID облачного ноутбука')
                if self.state.get('mode') in PLANNED_MODES:
                    self.state.update(message='Импорт не подтверждён. Проверяем список ноутбуков через форму 1.')
                else:
                    self.state.update(status='paused', message='Неопределённый результат запуска: требуется проверка формы 1')
            elif job.get('status') in END_STATUSES | {'error'} or solution:
                test['state'] = 'finished' if job.get('session_released') else 'releasing'
                if job.get('session_released'):
                    test.update(finished=self.clock(), released=True)
                    self.state['listed_at'] = 0
                test.setdefault('calculation_finished', self.clock())
                test['outcome'] = test['classification']
                test['terminal_error'] = job.get('error') or report.get('error')
            elif action == 'result' and test.pop('limit_poll', False):
                test.update(state='releasing', outcome='session_wall_time_limit')
            elif (action == 'result' and self.state.get('mode') in PLANNED_MODES
                  and job.get('status') == 'awaiting_heartbeat' and not receipt.get('error')
                  and not job.get('drive_received_utc') and self.clock() - test['created'] >= FIRST_WRITE_TIMEOUT):
                # Successful reads of the run's Drive folder still yield only registration.
                test.update(state='releasing', outcome='no_first_drive_write')
                self.event('no_first_drive_write', test_id=test['test_id'], age_seconds=self.clock()-test['created'])
            else:
                test['state'] = 'running'
                test['consecutive_errors'] = test.get('consecutive_errors', 0) + 1 if receipt.get('error') else 0
                received = job.get('drive_received_utc')
                if received:
                    from datetime import datetime
                    try:
                        last_write = datetime.fromisoformat(received.replace('Z','+00:00')).timestamp()
                    except (TypeError, ValueError):
                        last_write = test['created']
                else:
                    last_write = test['created']
                if action == 'result' and self.clock() - last_write >= 1200:
                    test['check_cloud_presence'] = True
            test['last_receipt'] = str(receipt_path)
            atomic_json(self.directory / (test['test_id'] + '.json'), test)
        self.state['pending'] = None
        self.reconcile_local_launches()
        self.export()
        self.persist()
        return True

    def plan_next(self):
        tests = self.state['jobs']
        for previous in tests:
            if previous.get('replay_pending'):
                fields = ('case_number','cube','beam','symmetry','inverse','requested_depth','lane',
                          'estimated_seconds','cuda_diagnostics','trace_reference','repair_stage','purpose')
                candidate = {name:previous[name] for name in fields if name in previous}
                return dict(candidate, replay_of=previous['test_id'], attempt=previous.get('attempt',1)+1)
        if self.state.get('mode') == 'repair':
            remaining = self.state['deadline'] - self.clock() - CLEANUP_MARGIN
            for stage in (0,1,2):
                active_stage = False
                for case in self.state['plan']:
                    if case['repair_stage'] != stage:
                        continue
                    previous = [t for t in tests if t['case_number'] == case['case_number']]
                    if any(t.get('state') != 'finished' for t in previous):
                        active_stage = True
                    elif case.get('estimated_seconds',0) > remaining:
                        continue
                    elif not previous:
                        candidate = dict(case, attempt=1)
                        if (not candidate.get('cuda_diagnostics') and len(tests) >= 2
                                and all(t.get('outcome') == 'cuda_illegal_access' for t in tests[-2:])):
                            candidate['cuda_diagnostics'] = 'sync'
                        return candidate
                    elif (len(previous) == 1 and previous[0].get('released')
                          and previous[0].get('outcome') in RETRYABLE
                          and not met_target(previous[0], previous[0].get('baseline_length'))):
                        return dict(case, attempt=2)
                if active_stage:
                    return None
            if remaining < 660:
                return None
            # Use spare time for additional orientations. Derive choices from
            # the immutable experiment seed and saved cases after every restart.
            rng = random.Random(self.state['seed'] + len(self.state['plan']))
            tried = {(c['cube'], c['beam'], c['symmetry'], c['inverse']) for c in self.state['plan']}
            lengths = load_lengths(self.directory / 'baseline.csv')
            long_cubes = sorted(k for k,n in lengths.items() if n >= 111)
            rng.shuffle(long_cubes)
            candidates = [(k, 3_500_000, min(120, lengths[k]+8)) for k in long_cubes]
            candidates += [(68, 3_500_000, 45), (68, 1_100_000, 45), (45, 200_000, 18)]
            for cube, beam, depth in candidates:
                estimate = math.ceil(600 + self.state['estimated_seconds_per_million'] * beam / 1e6 * (depth-3) * 1.18)
                if estimate > min(remaining, SESSION_LIMIT-CLEANUP_MARGIN):
                    continue
                orientations = [(s,inv) for s in range(48) for inv in (False,True) if (cube,beam,s,inv) not in tried]
                if orientations:
                    symmetry, inverse = rng.choice(orientations)
                    extra = dict(case_number=max((c['case_number'] for c in self.state['plan']),default=0)+1,
                        cube=cube, beam=beam, symmetry=symmetry, inverse=inverse, requested_depth=depth,
                        estimated_seconds=estimate, repair_stage=2, purpose='Дополнительная проверка до окончания серии',
                        baseline_length=lengths[cube], lane='reserve', trace_reference=cube==68)
                    self.state['baseline_lengths'][str(cube)] = lengths[cube]
                    self.state['plan'].append(extra)
                    return dict(extra,attempt=1)
            return None
        if self.state.get('mode') in {'diagnostics', 'tpu_controls'}:
            scheduled = {t['case_number'] for t in tests}
            remaining = self.state['deadline'] - self.clock() - CLEANUP_MARGIN
            cap = min(remaining, SESSION_LIMIT-CLEANUP_MARGIN)
            step_seconds = [r['seconds'] for t in tests for r in t.get('layers', [])
                            if r.get('full_beam') and r.get('depth',0) >= 7 and r.get('seconds',0) > 0]
            measured = statistics.median(step_seconds) if self.state['mode']=='tpu_controls' and step_seconds else None
            for case in self.state['plan']:
                if case['case_number'] in scheduled:
                    continue
                estimate = case.get('estimated_seconds',0)
                if measured is not None:
                    estimate = math.ceil(600 + measured * (case['requested_depth'] - 3))
                if estimate <= cap:
                    return dict(case, estimated_seconds=estimate)
            return None
        if self.state.get('mode') == 'single':
            config = self.state['single']
            if tests:
                return None
            return dict(cube=config['cube'], beam=config['beam'], requested_depth=config['depth'],
                        inverse=config.get('inverse',False),
                        cuda_diagnostics=config.get('cuda_diagnostics','off'), lane='single')
        tried = {(t['cube'], t['beam']) for t in tests}
        calibrations = [t for t in tests if t['cube'] == 57]
        if (57, 200000) not in tried:
            return dict(cube=57, beam=200000, lane='beam')
        if (67, 3700000) not in tried:
            return dict(cube=67, beam=3700000, lane='depth')
        for beam in (1000000, 3700000):
            if (57, beam) not in tried:
                return dict(cube=57, beam=beam, lane='beam')
        qualified = [t for t in calibrations if t.get('outcome') == 'solution'
                     and any(r['full_beam'] for r in t.get('layers', []))]
        good_beam = max((t['beam'] for t in qualified), default=None)
        if not good_beam:
            # Retrying infrastructure errors is bounded and preserves failures.
            retry = next((t for t in calibrations if t.get('released') and t.get('outcome') in
                          {'cloud_unavailable', 'gpu_unavailable', 'unclassified_error'} and
                          sum(x['cube'] == t['cube'] and x['beam'] == t['beam'] for x in tests) < 2), None)
            return dict(cube=57, beam=retry['beam'], lane='retry') if retry else None
        # Keep one capacity experiment and one long-path experiment active.
        if not any(t.get('state') != 'finished' and t['lane'] == 'depth' for t in tests):
            for cube in (67, 94, 124, 145):
                if (cube, good_beam) not in tried:
                    return dict(cube=cube, beam=good_beam, lane='depth')
        if any(t.get('state') != 'finished' and t['lane'] in {'beam', 'retry'} for t in tests):
            return None
        failures = [t['beam'] for t in calibrations if t.get('outcome') == 'resource_limit' and t['beam'] > good_beam]
        high = min(failures) if failures else None
        candidate = (int((good_beam + high) / 2 / 100000) * 100000 if high else good_beam * 2)
        if candidate <= good_beam or high and high - good_beam <= max(100000, good_beam * .05):
            return None
        if (57, candidate) not in tried:
            return dict(cube=57, beam=candidate, lane='beam')
        # A failed service/runtime is retried once without labelling it an OOM.
        previous = [t for t in calibrations if t['beam'] == candidate]
        if len(previous) < 2 and all(t.get('released') for t in previous):
            return dict(cube=57, beam=candidate, lane='retry')
        return None

    def begin_resume(self):
        self.state.update(status='resuming',user_stopped=False,listing=None,listed_at=0,next_list=0,
                          resume_cleanup=dict(started=self.clock(),next_check=self.clock(),
                                              inventory_valid=False,closed_ids=[],running_ids=[],read_jobs=[]),
                          message='Продолжение: проверяем Drive и освобождаем все сессии Molab.')
        self.state.pop('finished',None)
        self.event('resume_requested')
        self.persist()

    def continue_cleanup(self):
        cleanup = self.state['resume_cleanup']
        now = self.clock()
        if now < cleanup['next_check']:
            return
        if not cleanup['inventory_valid']:
            self.request('list')
            return
        active = [test for test in self.state['jobs'] if test['state'] != 'finished']
        for test in active:
            job = test.get('job') or {}
            if job.get('key') and job.get('drive_key') and test['test_id'] not in cleanup['read_jobs']:
                cleanup['read_jobs'].append(test['test_id'])
                self.request('result',test)
                return
        if cleanup['running_ids']:
            self.request('shutdown', target_id=cleanup['running_ids'][0])
            return
        for test in active:
            job = test.get('job') or {}
            # A complete inventory with zero sessions is the release barrier.
            # Keep a verified GPU outcome; reschedule anything interrupted by
            # this user-requested cleanup with the original solver parameters.
            report = self.report_for(job)
            completed = bool(verified_path(test)) or report.get('status') in END_STATUSES - {'stopped'}
            if completed:
                test['outcome'] = classify(job,report)
            else:
                test.update(outcome='interrupted_on_resume', replay_pending=True)
            test.update(state='finished',released=True,finished=now)
            atomic_json(self.directory / (test['test_id']+'.json'),test)
        self.event('resume_inventory_empty', closed_ids=cleanup['closed_ids'])
        self.state.pop('resume_cleanup')
        self.state.update(status='running', message='Все сессии освобождены. Продолжаем серию.')
        self.export()
        self.persist()

    def tick(self):
        while not self.commands.empty():
            action, value = self.commands.get_nowait()
            if action == 'start':
                self.start(**value)
            elif action == 'resume' or (action == 'pause' and self.state.get('status') == 'paused'):
                status = self.state.get('status')
                if status in {'paused','stopping'} or (status == 'finished' and self.state.get('user_stopped')):
                    margin = CLEANUP_MARGIN if self.state.get('mode') in PLANNED_MODES else 0
                    if self.clock() < self.state['deadline'] - margin:
                        self.begin_resume()
                    else:
                        self.state['message'] = 'Время этой серии истекло. Для новых тестов запустите новую серию.'
                        self.persist()
            elif self.state.get('status') in {'running','resuming'} and action == 'pause':
                self.state.update(status='paused',message='Пауза новых испытаний.')
                self.state.pop('resume_cleanup',None)
                self.event('paused')
                self.persist()
            elif self.state and action == 'stop':
                self.state.pop('resume_cleanup',None)
                self.state.update(status='stopping', user_stopped=True, message='Сохраняем результаты и завершаем только свои сессии')
                self.persist()
        if not self.state or self.state['status'] == 'finished':
            return
        self.reconcile_local_launches()
        if self.state['status'] == 'finished':
            return
        now = self.clock()
        diagnostics = self.state.get('mode') in PLANNED_MODES
        cleanup_at = self.state['deadline'] - (CLEANUP_MARGIN if self.state.get('mode') in {'diagnostics','repair'} else 0)
        if now >= cleanup_at and self.state['status'] != 'stopping':
            self.state.pop('resume_cleanup',None)
            self.state.update(status='stopping', message='Время испытания истекло. Получаем последние результаты своих сессий.')
            self.persist()
        if self.consume():
            return
        if self.state.get('status') == 'resuming':
            self.continue_cleanup()
            return
        active = [t for t in self.state['jobs'] if t.get('state') != 'finished']
        if diagnostics or self.state.get('mode') == 'single':
            for test in active:
                if (test['state']=='running' and test.get('job',{}).get('id') and
                    now - test['created'] >= SESSION_LIMIT-CLEANUP_MARGIN):
                    test['limit_poll'] = True
                    self.event('session_final_read',test_id=test['test_id'],age_seconds=now-test['created'])
                    self.request('result', test)
                    return
        if self.state.get('mode') == 'single' and self.state['jobs'] and not active and not any(
                t.get('replay_pending') for t in self.state['jobs']):
            outcome = self.state['jobs'][-1].get('outcome', 'нет подтверждённого результата')
            self.state.update(status='finished', finished=now, message=f'Одиночный тест завершён. Результат: {outcome}.')
            self.event('single_test_finished')
            self.export()
            self.persist()
            return
        releasing = next((t for t in active if t['state'] == 'releasing' and now >= t.get('next_shutdown', 0)), None)
        if releasing:
            self.request('shutdown', releasing)
            return
        stopping = self.state['status'] == 'stopping'
        if stopping:
            for test in active:
                if test['state'] == 'releasing':
                    continue
                if not test.get('job', {}).get('id'):
                    continue
                if not test.get('stop_polled'):
                    test['stop_polled'] = True
                    self.request('result', test)
                else:
                    test['state'] = 'releasing'
                    test.setdefault('outcome', 'time_budget_exhausted')
                    self.request('shutdown', test)
                return
            if active:
                if any(t['state']=='uncertain_launch' for t in active):
                    self.state['message'] = 'Проверка завершения импорта через форму 1'
                    can_reconcile = diagnostics or any((t.get('job') or {}).get('title') for t in active)
                    if can_reconcile and now >= self.state.get('next_list', 0):
                        self.request('list')
                    else:
                        self.refresh_snapshot()
                return
            self.state.update(status='finished', finished=now, message=self.state.get('halt_reason', 'Испытание закончено. Результаты сохранены.'))
            self.event('finished')
            self.export()
            self.persist()
            return
        due = next((t for t in active if t['state'] == 'running' and now >= t.get('next_poll', 0)), None)
        if due:
            self.request('result', due)
            return
        if self.state['status'] == 'paused':
            return
        if diagnostics:
            if any(t.get('check_cloud_presence') or t['state']=='uncertain_launch' for t in active) and now >= self.state.get('next_list', 0):
                self.request('list')
                return
        if len(active) >= MAX_RUNNING:
            return
        candidate = self.plan_next()
        if not candidate:
            if self.state.get('mode') == 'tpu_controls' and not active:
                self.state['message'] = 'До завершения серии недостаточно времени для ещё одного полного расчёта.'
                self.refresh_snapshot()
                return
            if diagnostics and not active:
                if self.state['mode'] == 'repair':
                    left = len(repair_plan(self.state, {int(k):v for k,v in self.state['baseline_lengths'].items()}))
                    message = f'Повтор завершён. Целевая длина ещё не подтверждена для {left} из {len(self.state["plan"])} случаев.'
                else:
                    message = 'Оставшиеся пары не помещаются до завершения ночной проверки.'
                self.state.update(status='finished',finished=now,message=message)
                self.export()
                self.persist()
                return
            self.state['message'] = 'Ожидаем подтверждённые результаты для следующего теста'
            self.refresh_snapshot()
            return
        if self.state['listed_at'] < now - 30:
            if now >= self.state.get('next_list', 0) or self.state['listing'] is not None:
                self.request('list')
            return
        running = self.state['listing']['running']
        running_ids = {str(r.get('id', r)) if isinstance(r, dict) else str(r) for r in running}
        owned_unlisted = sum(t.get('job', {}).get('id') not in running_ids for t in active)
        if len(running) + owned_unlisted >= MAX_RUNNING:
            self.state['message'] = 'Ожидаем освобождения единственной сессии Molab'
            self.state['listed_at'] = 0
            self.state['listing'] = None
            self.refresh_snapshot()
            return
        test = dict(candidate, test_id=uuid.uuid4().hex, symmetry=candidate.get('symmetry', self.state['symmetry']),
                    baseline_length=self.state.get('baseline_lengths',{}).get(str(candidate['cube']),self.lengths.get(candidate['cube'])),
                    created=now, state='launching', layers=[], completed_layers=None)
        test.setdefault('requested_depth', 120)
        if candidate.get('replay_of'):
            self.lookup(candidate['replay_of'])['replay_pending'] = False
        self.state['jobs'].append(test)
        self.state['listed_at'] = 0
        self.request('launch', test)

    def export(self):
        if not self.directory:
            return
        columns = ['test_id', 'case_number', 'cube', 'beam', 'effective_beam', 'symmetry', 'inverse', 'requested_depth','improvement','verified_path',
                   'baseline_length', 'found_length', 'verified', 'outcome', 'status', 'native_search_seconds',
                   'notebook_seconds', 'measured_max_frontier_from_depth7', 'full_layers_from_depth7',
                   'mean_full_layer_seconds', 'median_full_layer_seconds', 'completed_layers','last_completed_depth','drive_received_utc','native_reason','url', 'error']
        lines, layers = [], []
        for test in self.state.get('jobs', []):
            job = test.get('job', {})
            solution = test.get('verified_solution') or job.get('solution') or {}
            measured = test.get('layers', [])
            full = [r['seconds'] for r in measured if r['full_beam']]
            case = job.get('current_case') or {}
            report = self.report_for(job)
            attempts = report.get('attempts') or []
            lines.append(dict(test_id=test['test_id'], case_number=test.get('case_number'),cube=test['cube'], beam=test['beam'],
                completed_layers=test.get('completed_layers'),
                inverse=test.get('inverse', False), last_completed_depth=case.get('last_completed_depth',max((r['depth'] for r in measured), default=None)),
                drive_received_utc=job.get('drive_received_utc'),native_reason=(attempts[-1].get('stop_reason') or attempts[-1].get('search_outcome') or attempts[-1].get('error')) if attempts else report.get('stop_reason'),
                effective_beam=job.get('effective_beam') or solution.get('effective_beam'), symmetry=test['symmetry'],
                requested_depth=test.get('requested_depth', 120), baseline_length=test['baseline_length'], found_length=solution.get('length'),
                verified=solution.get('verified', False), improvement=test.get('improvement'), verified_path=solution.get('path') if solution.get('verified') else None,
                outcome=test.get('outcome', test.get('classification')),
                status=test['state'], native_search_seconds=solution.get('search_seconds') or job.get('search_seconds'),
                notebook_seconds=job.get('elapsed_seconds'),
                measured_max_frontier_from_depth7=max((r['frontier_size'] for r in measured), default=None),
                full_layers_from_depth7=len(full), mean_full_layer_seconds=statistics.mean(full) if full else None,
                median_full_layer_seconds=statistics.median(full) if full else None,
                url=job.get('url', ''), error=test.get('terminal_error') or test.get('last_error') or job.get('error','')))
            layers.extend(dict(test_id=test['test_id'], cube=test['cube'], requested_beam=test['beam'], **r) for r in measured)
        tables = [('results.csv', columns, lines), ('layers.csv',
            ['test_id','cube','requested_beam','rank','depth','frontier_size','started_utc','ended_utc','seconds','effective_beam','full_beam'], layers)]
        if self.state.get('mode') in {'diagnostics', 'repair'}:
            pairs = {}
            for case in self.state['plan']:
                group = (case['cube'],case['beam'],case['requested_depth'],case['symmetry'])
                pairs[group] = dict(zip(('cube','beam','depth','symmetry'),group))
            for row in lines:
                group = (row['cube'],row['beam'],row['requested_depth'],row['symmetry'])
                direction = 'reverse' if row['inverse'] else 'forward'
                pair = pairs[group]
                pair['baseline'] = row['baseline_length']
                pair[direction+'_outcome'] = row['outcome'] or row['status']
                pair[direction+'_length'] = row['found_length']
                pair[direction+'_seconds'] = row['native_search_seconds']
                pair[direction+'_url'] = row['url']
            cols = ['cube','beam','depth','symmetry','baseline','forward_outcome','reverse_outcome',
                    'forward_length','reverse_length','forward_seconds','reverse_seconds','forward_url','reverse_url']
            tables.append(('pairs.csv',cols,pairs.values()))
        for name, headers, rows in tables:
            buffer = io.StringIO(newline='')
            writer = csv.DictWriter(buffer, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
            target = self.directory / name
            temporary = target.with_suffix('.tmp')
            temporary.write_text('\ufeff' + buffer.getvalue(), encoding='utf-8')
            temporary.replace(target)


def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'


def scheduled_action(action, at, now):
    if at is None:
        return action
    if at <= now:
        return f'{action} · сейчас'
    stamp = datetime.fromtimestamp(at, MOSCOW).strftime('%H:%M:%S МСК')
    return f'{action} · через {duration(math.ceil(at - now))}, в {stamp}'


def wait_display(snapshot, now):
    """Describe the scheduler's persisted timers, without changing its queue."""
    stage = snapshot.get('status')
    if not stage:
        return 'Форма готова к запуску.', 'Выберите тест.'
    if stage == 'finished':
        return snapshot.get('message') or 'Испытание завершено.', 'Результаты доступны в таблице.'
    if stage == 'error':
        return snapshot.get('message', 'Ошибка контроллера.'), 'Проверьте сообщение и параметры запуска.'
    active = [t for t in snapshot.get('jobs', ()) if t.get('state') != 'finished']
    pending = snapshot.get('pending') or {}
    cleanup = snapshot.get('resume_cleanup') or {}
    prefix = {'paused':'Пауза новых запусков. ', 'resuming':'Подготовка к продолжению. ',
              'stopping':'Завершение серии. '}.get(stage,'')
    if pending:
        test = next((t for t in active if t.get('test_id') == pending.get('test_id')), {})
        target = f" · кубик {test['cube']}" if 'cube' in test else ''
        if not target and pending.get('target_id'):
            target = ' · ' + str(pending['target_id'])
        action = {'list':'проверить работающие сессии Molab', 'result':'прочитать отчёт с Google Drive',
                  'shutdown':'завершить сессию Molab', 'launch':'запустить ноутбук через форму 1'}.get(
                      pending.get('action'), 'выполнить команду формы 1') + target
        reason = pending.get('retry_reason') or ''
        retry_at = pending.get('next_retry')
        if reason:
            return prefix + str(reason), scheduled_action('Повторить операцию: ' + action, retry_at, now)
        started = pending.get('created', now)
        wait = f'Форма 1 выполняет команду «{action}». Прошло {duration(now - started)}.'
        if retry_at:
            return prefix + wait, scheduled_action('Проверка зависшего запроса при отсутствии ответа', retry_at, now)
        return prefix + wait, 'Проверить ответ формы 1 на следующем такте (через 1 с).'
    if stage == 'resuming':
        remaining = cleanup.get('running_ids') or []
        if cleanup.get('inventory_valid') and remaining:
            return (prefix + f'Работающих сессий в очереди на остановку: {len(remaining)}.',
                    scheduled_action('Остановить сессию ' + str(remaining[0]), cleanup.get('next_check',now), now))
        return (prefix + 'Ожидаю подтверждения списка сессий.',
                scheduled_action('Проверить работающие сессии Molab', cleanup.get('next_check',now), now))
    tasks = []
    for test in active:
        cube = f"Кубик {test.get('cube', '—')}."
        if test.get('state') == 'releasing':
            reason = test.get('last_error') or 'Ожидаю подтверждения остановки сессии Molab.'
            tasks.append((test.get('next_shutdown',now), f'{cube} {reason}', f"Остановить сессию кубика {test.get('cube')}"))
        elif test.get('state') == 'running':
            case = (test.get('job') or {}).get('current_case') or {}
            depth = case.get('depth')
            reason = 'Ожидаю отчёт ноутбука на Google Drive.'
            if depth is not None:
                reason += f' Последний полученный индекс слоя {depth}.'
            tasks.append((test.get('next_poll',now), f'{cube} {reason}', f"Прочитать Google Drive для кубика {test.get('cube')}"))
        if test.get('check_cloud_presence') or test.get('state') == 'uncertain_launch':
            tasks.append((snapshot.get('next_list',now), f'{cube} Проверяю наличие сессии в Molab.',
                          'Проверить список сессий'))
    if tasks:
        at, reason, action = min(tasks, key=lambda task:task[0])
        return prefix + reason, scheduled_action(action, at, now)
    if stage == 'paused':
        return 'Испытания приостановлены.', 'После «Продолжить» форма проверит и остановит сессии Molab перед запуском теста.'
    if stage == 'stopping':
        return 'Завершаю серию.', 'Проверить завершение сессий и сохранить результаты.'
    next_list = snapshot.get('next_list', 0)
    if snapshot.get('listing') is None:
        reason = snapshot.get('message') or ''
        if reason.startswith('Форма 1:') or not reason:
            reason = 'Ожидаю список работающих сессий Molab.'
        return reason, scheduled_action('Запросить список сессий через форму 1', next_list, now)
    return 'Проверяю очередь испытаний.', 'Выбрать следующий тест на ближайшем такте (через 1 с).'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--start', action='store_true', help='Нажать кнопку испытания на 24 часа')
    action.add_argument('--test', action='store_true', help='Нажать кнопку одиночного теста через форму 1')
    action.add_argument('--diagnostics', action='store_true', help='Ночная проверка на 10 часов, до полудня МСК')
    action.add_argument('--tpu-controls', action='store_true', help='10 часов на кубики с длинами 40, 100 и 110')
    action.add_argument('--repair', type=Path, help='Повторить неудачные тесты из завершённой серии state.json')
    action.add_argument('--resume', action='store_true', help='Продолжить текущую серию после освобождения сессий')
    action.add_argument('--stop', action='store_true', help='Завершить серию и освободить её сессию до нового теста')
    parser.add_argument('--inverse', action='store_true', help='Обратный поиск для одиночного теста')
    parser.add_argument('--cuda-diagnostics', choices=('off','sync','memcheck'), default='off')
    parser.add_argument('--seed', type=int, help='Начальное число для повторения порядка симметрий')
    parser.add_argument('--baseline', type=Path, default=CURRENT_CSV, help='Текущий CSV с путями для ночной проверки')
    parser.add_argument('--first-form', type=Path, default=FIRST)
    parser.add_argument('--symmetry', type=int, choices=range(48), default=3)
    parser.add_argument('--cube', type=int, default=39)
    parser.add_argument('--beam', type=int, default=200000)
    parser.add_argument('--depth', type=int, default=120)
    parser.add_argument('--request-id', help='UUID запроса для идемпотентного нажатия кнопки')
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    requests = DATA / 'form-requests'
    requests.mkdir(exist_ok=True)
    request_id = args.request_id or uuid.uuid4().hex
    try:
        if uuid.UUID(request_id).hex != request_id:
            raise ValueError()
    except ValueError:
        parser.error('--request-id должен быть UUID из 32 шестнадцатеричных символов')
    requested = args.start or args.test or args.diagnostics or args.tpu_controls or args.repair or args.resume or args.stop
    if requested:
        if (requests / (request_id + '.receipt')).exists():
            return
        atomic_json(requests / (request_id + '.json'), dict(start=args.start, test=args.test,
            diagnostics=args.diagnostics, tpu_controls=args.tpu_controls, seed=args.seed, baseline=str(args.baseline),
            repair=str(args.repair.resolve()) if args.repair else None, resume=args.resume, stop=args.stop,
            inverse=args.inverse, cuda_diagnostics=args.cuda_diagnostics,
            cube=args.cube, beam=args.beam, depth=args.depth, symmetry=args.symmetry))
    # One local supervisor prevents duplicated schedules after double-clicks.
    import msvcrt
    lock = (DATA / 'benchmark.lock').open('a+b')
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        if not requested:
            messagebox.showinfo('Cube555', 'Форма испытаний уже открыта. Используйте существующее окно.')
        return
    controller = Controller(first=args.first_form)
    controller.reconcile_local_launches()
    root = tk.Tk()
    root.title('Cube555 — испытания Molab (форма 2)')
    root.geometry('1180x840')
    root.minsize(1000, 720)
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', font=('Segoe UI', 10))
    style.configure('TButton', padding=(12, 8))
    outer = ttk.Frame(root, padding=20)
    outer.pack(fill='both', expand=True)
    ttk.Label(outer, text='Автономные испытания Molab', font=('Segoe UI', 20, 'bold')).pack(anchor='w')
    ttk.Label(outer, text='Одна облачная сессия. Опрос формы 1 каждые 5 минут.',
              font=('Segoe UI', 11)).pack(anchor='w', pady=(4, 12))
    controls = ttk.Frame(outer)
    controls.pack(fill='x')
    diagnostic_options = dict(diagnostics=True, baseline=str(args.baseline), seed=args.seed)
    diagnostic_button = ttk.Button(controls, text='Ночная проверка · 10 часов',
                                   command=lambda: controller.commands.put(('start',dict(diagnostic_options))))
    diagnostic_button.pack(side='left', padx=(0,8))
    tpu_options = dict(tpu_controls=True, baseline=str(args.baseline))
    tpu_button = ttk.Button(controls, text='40 / 100 / 110 · 10 часов',
                            command=lambda: controller.commands.put(('start',dict(tpu_options))))
    tpu_button.pack(side='left', padx=(0,8))
    ttk.Label(controls, text='Симметрия (0–47)').pack(side='left')
    symmetry = tk.StringVar(value=str(args.symmetry))
    choose = ttk.Combobox(controls, textvariable=symmetry, values=list(range(48)), width=5, state='readonly')
    choose.pack(side='left', padx=8)
    start = ttk.Button(controls, text='Запустить на 24 часа', command=lambda: controller.commands.put(
        ('start', {'symmetry': int(symmetry.get()), 'hours': 24})))
    start.pack(side='left', padx=5)
    def toggle_pause():
        controller.commands.put(('pause' if controller.snapshot.get('status') in {'running','resuming'} else 'resume', None))
    pause = ttk.Button(controls, text='Пауза новых тестов', command=toggle_pause)
    pause.pack(side='left', padx=5)
    stop = ttk.Button(controls, text='Завершить испытание', command=lambda: controller.commands.put(('stop', None)))
    stop.pack(side='left', padx=5)
    single_controls = ttk.Frame(outer)
    single_controls.pack(fill='x', pady=(12, 0))
    single_fields = {}
    for name, caption, default, width in [('cube','Кубик',args.cube,8), ('beam','Луч',args.beam,16), ('depth','Глубина',args.depth,8)]:
        ttk.Label(single_controls, text=caption).pack(side='left', padx=(0, 5))
        variable = tk.StringVar(value=str(default))
        field = ttk.Entry(single_controls, textvariable=variable, width=width)
        field.pack(side='left', padx=(0, 14))
        single_fields[name] = (variable, field)
    inverse_value = tk.BooleanVar(value=args.inverse)
    cuda_value = tk.StringVar(value=args.cuda_diagnostics)
    ttk.Checkbutton(single_controls, text='Реверс', variable=inverse_value).pack(side='left', padx=(0,5))
    ttk.Label(single_controls, text='CUDA').pack(side='left')
    ttk.Combobox(single_controls, textvariable=cuda_value, values=('off','sync','memcheck'),
                 state='readonly', width=8).pack(side='left', padx=(3,8))
    def single_test():
        try:
            single = {name:int(variable.get().strip()) for name,(variable, _) in single_fields.items()}
            single.update(inverse=inverse_value.get(), cuda_diagnostics=cuda_value.get())
            controller.commands.put(('start', {'symmetry':int(symmetry.get()), 'hours':24, 'single':single}))
        except ValueError:
            status.set('Кубик, луч и глубина должны быть целыми числами.')
    single_start = ttk.Button(single_controls, text='Запустить один тест', command=single_test)
    single_start.pack(side='left')
    def repair_run():
        target = repair_source.get() or str((controller.directory or DATA) / 'state.json')
        controller.commands.put(('start', {'repair':target, 'hours':10}))
    repair_source = tk.StringVar()
    repair_button = ttk.Button(single_controls, text='Повторить неудачные', command=repair_run)
    repair_button.pack(side='left', padx=8)
    status = tk.StringVar(value='Готово. При открытии новые тесты автоматически не создаются.')
    clock_text = tk.StringVar()
    next_action = tk.StringVar()
    summary = tk.StringVar(value='Измерения полного луча учитываются с шага 7 и только по фактическому frontier_size.')
    ttk.Label(outer, textvariable=status, wraplength=1100).pack(anchor='w', pady=(15, 4))
    ttk.Label(outer, textvariable=clock_text).pack(anchor='w')
    ttk.Label(outer, textvariable=next_action, wraplength=1090).pack(anchor='w')
    ttk.Label(outer, textvariable=summary, wraplength=1100).pack(anchor='w', pady=(8, 12))
    columns = ('cube', 'beam', 'phase', 'step', 'completed', 'frontier', 'length', 'seconds', 'poll')
    tree = ttk.Treeview(outer, columns=columns, show='headings', height=13)
    tree.tag_configure('improvement', background='#d4efda', foreground='#18532f')
    headings = ['Кубик', 'Лимит луча', 'Состояние', 'Шаг', 'Пройдено слоёв', 'Фактический луч', 'Длина', 'Время / с', 'След. опрос']
    for col, title in zip(columns, headings):
        tree.heading(col, text=title)
        tree.column(col, width=90 if col != 'phase' else 210, stretch=True)
    tree.pack(fill='both', expand=True)
    details = tk.Text(outer, height=9, wrap='word', font=('Consolas', 10))
    details.pack(fill='x', pady=(12, 0))
    footer = ttk.Frame(outer)
    footer.pack(fill='x', pady=(12, 0))
    ttk.Button(footer, text='Открыть форму 1', command=lambda: controller.invoke([])).pack(side='left')
    ttk.Button(footer, text='Открыть результаты', command=lambda: os.startfile(str(controller.directory or DATA))).pack(side='left', padx=8)
    ttk.Label(footer, text='В свёрнутом окне тесты идут. После закрытия откройте форму для продолжения.').pack(side='left')
    selected_detail = ''

    def render():
        nonlocal selected_detail
        snapshot = controller.snapshot
        now = time.time()
        active = snapshot.get('status') in {'running', 'paused', 'resuming', 'stopping'}
        repair_button.state(['disabled'] if active else ['!disabled'])
        tpu_button.state(['disabled'] if active else ['!disabled'])
        start.state(['disabled'] if active else ['!disabled'])
        diagnostic_button.state(['disabled'] if active else ['!disabled'])
        single_start.state(['disabled'] if active else ['!disabled'])
        choose.state(['disabled'] if active else ['!disabled'])
        for _, field in single_fields.values():
            field.state(['disabled'] if active else ['!disabled'])
        may_continue = snapshot.get('status') in {'paused','stopping'} or (snapshot.get('status')=='finished' and snapshot.get('user_stopped'))
        pause.state(['!disabled'] if active or may_continue else ['disabled'])
        stop.state(['!disabled'] if active else ['disabled'])
        pause.configure(text='Продолжить тесты' if may_continue else 'Пауза новых тестов')
        if snapshot:
            current_wait, next_step = wait_display(snapshot, now)
            status.set(current_wait)
            next_action.set(next_step)
            end = snapshot.get('finished', now)
            mode = {'single':'Одиночный тест', 'tpu_controls':'40 / 100 / 110 · 10 часов', 'repair':'Повтор неудачных тестов', 'diagnostics':'Ночная проверка до ' + datetime.fromtimestamp(snapshot['deadline'],MOSCOW).strftime('%H:%M МСК')}.get(snapshot.get('mode'),'Испытание на 24 часа')
            clock_text.set(f"{mode} · прошло {duration(end - snapshot['started'])} · осталось {duration(snapshot['deadline'] - now)}")
        jobs = snapshot.get('jobs', [])
        ids = {t['test_id'] for t in jobs}
        for row in tree.get_children():
            if row not in ids:
                tree.delete(row)
        for test in jobs:
            job = test.get('job', {})
            case = (job.get('current_case') or {})
            found = test.get('verified_solution') or job.get('solution') or {}
            tags = ('improvement',) if test.get('improvement',0)>0 else ()
            values = (test['cube'], f"{test['beam']:,}", test.get('outcome', job.get('status', test['state'])),
                case.get('depth', '—'), test['completed_layers'] if test.get('completed_layers') is not None else '—',
                case.get('frontier_size', '—'), found.get('length', '—'),
                round(job.get('elapsed_seconds') or 0, 1), duration(test.get('next_poll', now) - now) if test['state'] == 'running' else '—')
            if tree.exists(test['test_id']):
                tree.item(test['test_id'], values=values,tags=tags)
            else:
                tree.insert('', 'end', iid=test['test_id'], values=values,tags=tags)
        measured = [r for t in jobs for r in t.get('layers', []) if r.get('full_beam')]
        max_beam = max((r['frontier_size'] for r in measured), default=0)
        max_length = max(((t.get('job', {}).get('solution') or {}).get('length', 0) for t in jobs), default=0)
        summary.set(f'Подтверждённый полный луч с шага 7: {max_beam:,}  ·  Максимальная проверенная длина: {max_length}  ·  Измеренных полных шагов: {len(measured)}')
        selection = tree.selection()
        test = next((t for t in jobs if selection and t['test_id'] == selection[0]), jobs[-1] if jobs else None)
        if test:
            job = test.get('job', {})
            full = [r['seconds'] for r in test.get('layers', []) if r['full_beam']]
            text = (f"Кубик {test['cube']} · луч {test['beam']} · симметрия {test['symmetry']} · предел глубины {test.get('requested_depth', 120)}\n"
                    f"Исходная длина CSV: {test['baseline_length']}\n"
                    f"Среднее время полного шага ≥7: {statistics.mean(full):.3f} с\n" if full else
                    f"Кубик {test['cube']} · луч {test['beam']} · симметрия {test['symmetry']}\nПолных измеренных шагов ≥7 пока нет.\n")
            direction = 'обратное' if test.get('inverse') else 'прямое'
            text += f"Направление: {direction}. Исходная длина CSV: {test['baseline_length']}.\n{job.get('url', '')}\nGoogle Drive: {job.get('drive_url', '—')}\nОшибка: {test.get('last_error', 'нет')}\nПуть: {(job.get('solution') or {}).get('path', 'ещё не получен')}\n{job.get('last_log', '')[-3500:]}"
            if test.get('improvement',0)>0:
                text = f"Проверенное улучшение на {test['improvement']} ходов.\n" + text
            if text != selected_detail:
                selected_detail = text
                details.configure(state='normal')
                details.delete('1.0', 'end')
                details.insert('1.0', text)
                details.configure(state='disabled')
        if controller.commands.empty():
            pending = sorted(requests.glob('*.json'))
            if pending:
                request_path = pending[0]
                try:
                    command = read_json(request_path)
                    if command.get('stop'):
                        if active:
                            stop.invoke()
                        atomic_json(request_path.with_suffix('.receipt'),dict(ok=True,handler='stop_button'))
                        request_path.unlink(missing_ok=True)
                        root.after(1000, render)
                        return
                    if command.get('resume'):
                        if not may_continue:
                            raise ValueError('Серия уже выполняется или срок её продолжения истёк.')
                        pause.invoke()
                        atomic_json(request_path.with_suffix('.receipt'),dict(ok=True,handler='resume_button'))
                        request_path.unlink(missing_ok=True)
                        root.after(1000, render)
                        return
                    if active:
                        raise ValueError('Сначала завершите текущее испытание; новый запуск не создан.')
                    symmetry.set(str(int(command['symmetry'])))
                    if command.get('repair'):
                        repair_source.set(command['repair'])
                        repair_button.invoke()
                        repair_source.set('')
                        handler = 'repair_button'
                    elif command.get('tpu_controls'):
                        source = Path(command['baseline'])
                        tpu_control_plan(load_lengths(source))
                        tpu_options.update(baseline=str(source))
                        tpu_button.invoke()
                        handler = 'tpu_controls_button'
                    elif command.get('diagnostics'):
                        source=Path(command['baseline'])
                        load_lengths(source)
                        diagnostic_options.update(baseline=str(source),seed=command.get('seed'))
                        diagnostic_button.invoke()
                        handler = 'diagnostics_button'
                    elif command.get('test'):
                        inverse_value.set(command.get('inverse',False))
                        cuda_value.set(command.get('cuda_diagnostics','off'))
                        for name, (variable, _) in single_fields.items():
                            variable.set(str(int(command[name])))
                        single_start.invoke()
                        handler = 'single_test_button'
                    else:
                        start.invoke()
                        handler = 'benchmark_start_button'
                    atomic_json(request_path.with_suffix('.receipt'), dict(ok=True, handler=handler, parameters=command))
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    atomic_json(request_path.with_suffix('.receipt'), dict(ok=False, error=str(exc)))
                    status.set(str(exc))
                request_path.unlink(missing_ok=True)
        root.after(1000, render)

    def worker():
        while True:
            try:
                controller.tick()
            except Exception as exc:
                if controller.state:
                    controller.state['message'] = f'Ошибка контроллера: {type(exc).__name__}: {exc}'
                    controller.event('controller_error', error=repr(exc))
                    controller.persist()
                else:
                    controller.snapshot = {'status': 'error', 'message': str(exc), 'started': time.time(), 'deadline': time.time()}
            time.sleep(1)

    def copy_selection(event=None):
        if event is not None and event.widget is tree:
            rows = tree.selection()
            text = '\n'.join(['\t'.join(headings)] + ['\t'.join(map(str, tree.item(row,'values'))) for row in rows])
        else:
            text = '\n'.join([status.get(), clock_text.get(), summary.get(), details.get('1.0','end-1c')])
        root.clipboard_clear()
        root.clipboard_append(text)
        return 'break'
    tree.bind('<Control-c>',copy_selection)
    ttk.Button(footer,text='Скопировать отчёт',command=copy_selection).pack(side='left')
    root.protocol('WM_DELETE_WINDOW', root.destroy)
    # Consume queued button commands before the first scheduler tick on restart.
    # In particular an explicit Stop must precede dispatching another notebook.
    render()
    threading.Thread(target=worker, daemon=True).start()
    root.mainloop()


if __name__ == '__main__':
    main()
