"""Offline contract tests: no network, no processes, no cloud launches."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('benchmark', Path(__file__).with_name('cube555_benchmark.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.exe = self.directory / 'form1/Control.exe'
        self.exe.parent.mkdir()
        self.exe.write_bytes(b'not executed')
        self.now = 100000.
        self.invocations = []
        self.controller = module.Controller(first=self.exe, data=self.directory / 'state',
                                            clock=lambda: self.now, invoke=self.invocations.append)
        self.controller.start()

    def receipt(self, value):
        pending = self.controller.state['pending']
        module.atomic_json(self.controller.request_directory / (pending['id'] + '.receipt'), value)
        self.controller.tick()

    def list_and_launch(self):
        self.controller.tick()
        self.receipt({'listing': {'running': []}, 'handler': 'list_button'})
        self.controller.tick()
        self.assertEqual(self.controller.state['pending']['action'], 'launch')
        self.assertIn('--managed', self.invocations[-1])
        self.receipt({'job': {'id': 'nb_test', 'key': 'job_1', 'status': 'running'}, 'handler': 'start_button'})
        return self.controller.state['jobs'][0]

    def test_one_existing_cloud_session_prevents_launch(self):
        self.controller.tick()
        self.receipt({'listing': {'running': [{'id': 'other1'}]}})
        self.controller.tick()
        self.assertEqual(len(self.invocations), 1)
        self.assertFalse(self.controller.state['jobs'])
        self.now += 299
        self.controller.tick()
        self.assertEqual(len(self.invocations), 1)
        self.now += 1
        self.controller.tick()
        self.assertEqual(self.controller.state['pending']['action'], 'list')

    def test_inventory_failure_waits_one_minute_before_retry(self):
        self.controller.tick()
        self.receipt({'error':'dashboard timeout'})
        self.now += 59
        self.controller.tick()
        self.assertIsNone(self.controller.state['pending'])
        self.now += 1
        self.controller.tick()
        self.assertEqual(self.controller.state['pending']['action'], 'list')
        self.assertFalse(self.controller.state['jobs'])

    def test_only_exact_receipt_and_no_duplicate_launch_after_timeout(self):
        self.controller.tick()
        self.receipt({'listing': {'running': []}})
        self.controller.tick()
        request_id = self.controller.state['pending']['id']
        module.atomic_json(self.controller.request_directory / ('wrong.receipt'), {'job': {'id': 'wrong'}})
        self.now += 2000
        self.controller.tick()
        self.assertEqual(len(self.invocations), 3)
        self.assertEqual(self.invocations[-2], self.invocations[-1])
        self.assertEqual(self.controller.state['pending']['id'], request_id)
        self.assertEqual(len(self.controller.state['jobs']), 1)
        resumed = module.Controller(first=self.exe, data=self.directory / 'state', clock=lambda: self.now,
                                     invoke=self.invocations.append)
        resumed.tick()
        self.assertEqual(len(self.invocations), 3)

    def test_screen_busy_retries_same_launch_after_one_minute(self):
        self.controller.tick()
        self.receipt({'listing': {'running': []}})
        self.controller.tick()
        launch = self.invocations[-1]
        self.receipt(dict(error='Экран занят', code='screen_busy', retryable=True, retry_at=self.now+60))
        self.now += 59
        self.controller.tick()
        self.assertEqual(len(self.invocations), 2)
        self.now += 1
        self.controller.tick()
        self.assertEqual(len(self.invocations), 3)
        self.assertEqual(self.invocations[-1], launch)
        self.assertEqual(len(self.controller.state['jobs']), 1)

    def test_managed_results_are_polled_only_after_300_seconds(self):
        test = self.list_and_launch()
        self.controller.state['status'] = 'paused'
        self.now += 299
        self.controller.tick()
        self.assertIsNone(self.controller.state['pending'])
        self.now += 1
        self.controller.tick()
        self.assertEqual(self.controller.state['pending']['action'], 'result')
        self.assertEqual(self.controller.state['pending']['test_id'], test['test_id'])

    def test_deadline_polls_then_releases_only_owned_job(self):
        test = self.list_and_launch()
        self.now = self.controller.state['deadline'] + 1
        self.controller.tick()
        self.assertEqual(self.controller.state['pending']['action'], 'result')
        self.receipt({'job': {'id': 'nb_test', 'key': 'job_1', 'status': 'running'}})
        self.controller.tick()
        self.assertEqual(self.controller.state['pending']['action'], 'shutdown')
        self.assertEqual(self.invocations[-1][-1], 'job_1')
        self.receipt({'job': {'id': 'nb_test', 'key': 'job_1', 'status': 'stopped'}})
        self.controller.tick()
        self.assertEqual(self.controller.state['status'], 'finished')
        self.assertTrue((self.controller.directory / 'results.csv').is_file())

    def test_resource_failure_distinct_from_cloud_failure(self):
        self.assertEqual(module.classify({'error': 'HTTP 403 runtime unavailable'}), 'cloud_unavailable')
        self.assertEqual(module.classify({'error': 'CUDA out of memory'}), 'resource_limit')
        self.assertEqual(module.classify({'status': 'failed'}), 'native_error')

    def test_full_beam_requires_actual_frontier_and_depth_seven(self):
        rows = module.layer_samples({'effective_beam': 200704}, {'attempts': [{'effective_beam': 200704,
            'depth_timings': [{'depth': 6, 'seconds': 1., 'frontier_size': 200704},
                              {'depth': 7, 'seconds': 2., 'frontier_size': 1000},
                              {'depth': 8, 'seconds': 3., 'frontier_size': 200704},
                              {'depth': 9, 'seconds': 4.}]}]})
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]['full_beam'])
        self.assertTrue(rows[1]['full_beam'])
        self.assertEqual(rows[1]['seconds'], 3.)

    def test_initial_pair_and_adaptive_expansion_require_confirmation(self):
        self.assertEqual(self.controller.plan_next(), {'cube': 57, 'beam': 200000, 'lane': 'beam'})
        self.controller.state['jobs'] = [dict(cube=57, beam=200000, lane='beam', state='running')]
        self.assertEqual(self.controller.plan_next(), {'cube': 67, 'beam': 3700000, 'lane': 'depth'})

    def test_single_test_passes_parameters_waits_300_and_finishes_after_release(self):
        self.controller.state['status'] = 'finished'
        self.controller.start(symmetry=3, single={'cube':39, 'beam':200000, 'depth':30,
                                                'cuda_diagnostics':'sync','inverse':True})
        self.controller.tick()
        self.receipt({'listing': {'running': []}})
        self.controller.tick()
        arguments = self.invocations[-1]
        self.assertEqual(arguments[arguments.index('--cube')+1], '39')
        self.assertEqual(arguments[arguments.index('--beam')+1], '200000')
        self.assertEqual(arguments[arguments.index('--depth')+1], '30')
        self.assertEqual(arguments[arguments.index('--symmetry')+1], '3')
        self.assertEqual(arguments[arguments.index('--cuda-diagnostics')+1], 'sync')
        self.assertIn('--inverse', arguments)
        self.receipt({'job': {'id':'nb_39','key':'job39','status':'running','solution':None,'current_case':None}})
        self.now += 299
        self.controller.tick()
        self.assertIsNone(self.controller.state['pending'])
        self.assertEqual(len(self.controller.state['jobs']), 1)
        self.now += 1
        self.controller.tick()
        self.assertEqual(self.controller.state['pending']['action'], 'result')
        solution = {'length':5,'path':'r0.r0.r0.r0.r0','verified':True}
        self.receipt({'job': {'id':'nb_39','key':'job39','status':'complete','solution':solution}})
        self.controller.tick()
        self.assertEqual(self.controller.state['pending']['action'], 'shutdown')
        self.receipt({'job': {'id':'nb_39','key':'job39','status':'stopped','solution':solution}})
        self.controller.tick()
        self.assertEqual(self.controller.state['status'], 'finished')
        self.assertEqual(self.controller.state['jobs'][0]['outcome'], 'solution')
        self.assertEqual(self.controller.state['jobs'][0]['job']['solution']['length'], 5)
        self.assertEqual(len(self.controller.state['jobs']), 1)

    def uncertain_job(self, stage):
        job = dict(key='old_job', status='error', request_id='launch_1', error=stage+'\nFailure')
        test = dict(test_id='old_test', cube=57, beam=200000, symmetry=3, requested_depth=120,
                    baseline_length=23, nominal_length=23, lane='beam', state='uncertain_launch',
                    layers=[], launch_request_id='launch_1', job=job)
        directory = self.exe.parent / 'runs/old_job'
        module.atomic_json(directory / 'job.json', job)
        module.atomic_json(directory / 'error.json', {'stage':stage})
        self.controller.state.update(status='stopping', jobs=[test])
        return directory, test

    def test_preimport_failure_does_not_leave_stop_stuck_or_touch_cloud(self):
        _, test = self.uncertain_job('preparing')
        self.controller.tick()
        self.assertEqual(self.controller.state['status'], 'finished')
        self.assertEqual(test['outcome'], 'not_started')
        self.assertTrue(test['released'])
        self.assertFalse(self.invocations)

    def test_uncertain_import_is_not_assumed_absent(self):
        _, test = self.uncertain_job('importing')
        self.controller.tick()
        self.assertEqual(self.controller.state['status'], 'stopping')
        self.assertEqual(test['state'], 'uncertain_launch')
        self.assertFalse(self.invocations)

    def test_tpu_controls_validate_baselines_and_use_exact_ten_hour_budget(self):
        lengths = {76:40, 142:100, 994:110, 82:40, 166:100, 170:110}
        source = self.directory / 'submission.csv'
        source.write_text('initial_state_id,path\n' + ''.join(
            f"{cube},{'.'.join(['r0'] * lengths.get(cube, 1))}\n" for cube in range(1035)))
        self.controller.state['status'] = 'finished'
        self.controller.start(tpu_controls=True, baseline=source)
        state = self.controller.state
        self.assertEqual(state['deadline'] - state['started'], 36000)
        self.assertEqual(state['mode'], 'tpu_controls')
        self.assertEqual([v['cube'] for v in state['plan'][:6]], list(lengths))
        cases = {(v['cube'], v['symmetry'], v['inverse']) for v in state['plan']}
        self.assertEqual(len(cases), len(state['plan']))
        self.assertEqual((state['plan'][2]['symmetry'],state['plan'][2]['inverse']), (45,True))
        self.assertEqual(self.controller.plan_next()['beam'], 4194304)
        self.assertEqual(module.MAX_RUNNING, 1)
        with self.assertRaises(ValueError):
            module.tpu_control_plan({**lengths, 994:109})
        self.now = state['deadline'] - 300
        self.controller.tick()
        self.assertEqual(state['status'], 'running')
        self.assertFalse(self.invocations)
        self.now = state['deadline']
        self.controller.tick()
        self.assertEqual(state['status'], 'finished')

    def test_diagnostic_design_is_paired_repeatable_and_covers_48_frames(self):
        lengths = {45:11, 68:32, 71:36, 80:36}
        plan = module.diagnostic_plan(lengths, 1983)
        self.assertEqual(plan, module.diagnostic_plan(lengths, 1983))
        self.assertNotEqual(plan, module.diagnostic_plan(lengths, 1984))
        self.assertEqual((plan[0]['cube'],plan[0]['symmetry'],plan[0]['inverse']), (45,0,False))
        self.assertEqual({r['symmetry'] for r in plan if r['cube']==45}, set(range(48)))
        for f,r in zip(plan[::2],plan[1::2]):
            self.assertNotEqual(f['inverse'],r['inverse'])
            self.assertEqual([f[k] for k in ('cube','beam','symmetry','requested_depth')],
                             [r[k] for k in ('cube','beam','symmetry','requested_depth')])

    def test_numeric_counts_are_not_http_errors(self):
        self.assertEqual(module.classify({'status':'partial'},{'attempts':[{'depth':34,'wall_seconds':3502.403}]}), 'native_error')
        self.assertEqual(module.classify({'status':'failed'}, {'last_log':'cudaEventSynchronize: an illegal memory access was encountered'}), 'cuda_illegal_access')

    def test_historical_pair_and_moscow_noon(self):
        plan=module.diagnostic_plan({45:11,68:32,72:36},345)
        self.assertEqual((plan[2]['cube'],plan[2]['beam'],plan[2]['symmetry'],plan[2]['requested_depth']), (68,1100000,23,45))
        self.assertTrue(plan[2]['trace_reference'])
        for hour, end_hour in ((1,11),(2,12),(4,12)):
            start=module.datetime(2026,11,21,hour,tzinfo=module.MOSCOW)
            self.assertEqual(module.datetime.fromtimestamp(module.night_deadline(start.timestamp()),module.MOSCOW).hour,end_hour)

    def test_missing_length_36_uses_actual_nearest(self):
        plan=module.diagnostic_plan({45:11,68:32,79:41,74:35,76:38},1234)
        trial=next(t for t in plan if t['purpose'].startswith('Независимый кубик'))
        self.assertEqual((trial['cube'],trial['requested_depth']),(74,45))

    def test_session_release_before_five_hours_in_diagnostics(self):
        test=self.list_and_launch()
        self.controller.state.update(mode='diagnostics',plan=[dict(test)],deadline=self.now+36000)
        self.now=test['created']+module.SESSION_LIMIT-module.CLEANUP_MARGIN
        self.controller.tick()
        self.assertEqual(self.controller.state['pending']['action'], 'result')
        self.receipt({'job':dict(test['job'])})
        self.controller.tick()
        self.assertEqual(test['outcome'],'session_wall_time_limit')
        self.assertEqual(self.controller.state['pending']['action'],'shutdown')

    def test_saved_cloud_id_is_recovered_before_stop(self):
        directory, test = self.uncertain_job('importing')
        module.atomic_json(directory / 'job.json', {**test['job'], 'id':'nb_recovered'})
        self.controller.tick()
        self.assertEqual(test['job']['id'], 'nb_recovered')
        self.assertEqual(self.controller.state['pending']['action'], 'result')
        self.assertEqual(self.invocations[-1][-1], 'old_job')


if __name__ == '__main__':
    unittest.main()
