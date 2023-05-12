# Copyright 2018, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import contextlib

from absl import flags
from absl.testing import parameterized
import tensorflow as tf

from utils.optimizers import optimizer_utils

FLAGS = flags.FLAGS
TEST_CLIENT_FLAG_PREFIX = 'test_client'
TEST_SERVER_FLAG_PREFIX = 'test_server'


@contextlib.contextmanager
def flag_sandbox(flag_value_dict):

  def _set_flags(flag_dict):
    for name, value in flag_dict.items():
      FLAGS[name].value = value

  # Store the current values and override with the new.
  preserved_value_dict = {
      name: FLAGS[name].value for name in flag_value_dict.keys()
  }
  _set_flags(flag_value_dict)
  yield

  # Restore the saved values.
  for name in preserved_value_dict:
    FLAGS[name].unparse()
  _set_flags(preserved_value_dict)


def setUpModule():
  # Create flags here to ensure duplicate flags are not created.
  optimizer_utils.define_optimizer_flags(TEST_SERVER_FLAG_PREFIX)
  optimizer_utils.define_optimizer_flags(TEST_CLIENT_FLAG_PREFIX)
  optimizer_utils.define_lr_schedule_flags(TEST_SERVER_FLAG_PREFIX)
  optimizer_utils.define_lr_schedule_flags(TEST_CLIENT_FLAG_PREFIX)

# Create a list of `(test name, optimizer name flag value, optimizer class)`
# for parameterized tests.
_OPTIMIZERS_TO_TEST = [
    (name, name, cls)
    for name, cls in optimizer_utils._SUPPORTED_OPTIMIZERS.items()
]


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_create_optimizer_fn_from_flags_invalid_optimizer(self):
    FLAGS[f'{TEST_CLIENT_FLAG_PREFIX}_optimizer'].value = 'foo'
    with self.assertRaisesRegex(ValueError, 'not a valid optimizer'):
      optimizer_utils.create_optimizer_fn_from_flags(TEST_CLIENT_FLAG_PREFIX)

  def test_create_optimizer_fn_with_no_learning_rate(self):
    with flag_sandbox({f'{TEST_CLIENT_FLAG_PREFIX}_optimizer': 'sgd', f'{TEST_CLIENT_FLAG_PREFIX}_learning_rate': None}):
      with self.assertRaisesRegex(ValueError, 'Learning rate'):
        optimizer_utils.create_optimizer_fn_from_flags(TEST_CLIENT_FLAG_PREFIX)

  def test_create_optimizer_fn_from_flags_flags_set_not_for_optimizer(self):
    with flag_sandbox({f'{TEST_CLIENT_FLAG_PREFIX}_optimizer': 'sgd'}):
      # Set an Adam flag that isn't used in SGD.
      # We need to use `_parse_args` because that is the only way FLAGS is
      # notified that a non-default value is being used.
      bad_adam_flag = f'{TEST_CLIENT_FLAG_PREFIX}_adam_beta_1'
      FLAGS._parse_args(args=[f'--{bad_adam_flag}=0.5'], known_only=True)
      with self.assertRaisesRegex(
          ValueError,
          r'Commandline flags for .*\[sgd\].*\'test_client_adam_beta_1\'.*'):
        optimizer_utils.create_optimizer_fn_from_flags(TEST_CLIENT_FLAG_PREFIX)
      FLAGS[bad_adam_flag].unparse()

  @parameterized.named_parameters(_OPTIMIZERS_TO_TEST)
  def test_create_client_optimizer_from_flags(self, optimizer_name,
                                              optimizer_cls):
    commandline_set_learning_rate = 100.0
    with flag_sandbox({f'{TEST_CLIENT_FLAG_PREFIX}_optimizer': optimizer_name, f'{TEST_CLIENT_FLAG_PREFIX}_learning_rate': commandline_set_learning_rate}):

      custom_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      custom_optimizer = custom_optimizer_fn()
      self.assertIsInstance(custom_optimizer, optimizer_cls)
      self.assertEqual(custom_optimizer.get_config()['learning_rate'],
                       commandline_set_learning_rate)
      custom_optimizer_with_arg = custom_optimizer_fn(11.0)
      self.assertIsInstance(custom_optimizer_with_arg, optimizer_cls)
      self.assertEqual(
          custom_optimizer_with_arg.get_config()['learning_rate'], 11.0)

  @parameterized.named_parameters(_OPTIMIZERS_TO_TEST)
  def test_create_server_optimizer_from_flags(self, optimizer_name,
                                              optimizer_cls):
    commandline_set_learning_rate = 100.0
    with flag_sandbox({f'{TEST_SERVER_FLAG_PREFIX}_optimizer': optimizer_name, f'{TEST_SERVER_FLAG_PREFIX}_learning_rate': commandline_set_learning_rate}):
      custom_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags(
          TEST_SERVER_FLAG_PREFIX)
      custom_optimizer = custom_optimizer_fn()
      self.assertIsInstance(custom_optimizer, optimizer_cls)
      self.assertEqual(custom_optimizer.get_config()['learning_rate'],
                       commandline_set_learning_rate)
      custom_optimizer_with_arg = custom_optimizer_fn(11.0)
      self.assertIsInstance(custom_optimizer_with_arg, optimizer_cls)
      self.assertEqual(custom_optimizer_with_arg.get_config()['learning_rate'],
                       11.0)

  def test_create_constant_client_lr_schedule_from_flags(self):
    with flag_sandbox({f'{TEST_CLIENT_FLAG_PREFIX}_learning_rate': 3.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_schedule': 'constant'}):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 3.0, err=1e-5)
      self.assertNear(lr_schedule(1), 3.0, err=1e-5)
      self.assertNear(lr_schedule(105), 3.0, err=1e-5)
      self.assertNear(lr_schedule(1042), 3.0, err=1e-5)
    with flag_sandbox({f'{TEST_CLIENT_FLAG_PREFIX}_learning_rate': 3.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_schedule': 'constant', f'{TEST_CLIENT_FLAG_PREFIX}_lr_warmup_steps': 10}):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 0.3, err=1e-5)
      self.assertNear(lr_schedule(1), 0.6, err=1e-5)
      self.assertNear(lr_schedule(10), 3.0, err=1e-5)
      self.assertNear(lr_schedule(11), 3.0, err=1e-5)
      self.assertNear(lr_schedule(115), 3.0, err=1e-5)
      self.assertNear(lr_schedule(1052), 3.0, err=1e-5)

  def test_create_exp_decay_client_lr_schedule_from_flags(self):
    with flag_sandbox({f'{TEST_CLIENT_FLAG_PREFIX}_learning_rate': 3.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_schedule': 'exp_decay', f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_steps': 10, f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_rate': 0.1, f'{TEST_CLIENT_FLAG_PREFIX}_lr_staircase': True}):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 3.0, err=1e-5)
      self.assertNear(lr_schedule(3), 3.0, err=1e-5)
      self.assertNear(lr_schedule(10), 0.3, err=1e-5)
      self.assertNear(lr_schedule(19), 0.3, err=1e-5)
      self.assertNear(lr_schedule(20), 0.03, err=1e-5)

    with flag_sandbox({f'{TEST_CLIENT_FLAG_PREFIX}_learning_rate': 3.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_schedule': 'exp_decay', f'{TEST_CLIENT_FLAG_PREFIX}_lr_warmup_steps': 0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_steps': 10, f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_rate': 0.1, f'{TEST_CLIENT_FLAG_PREFIX}_lr_staircase': False}):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 3.0, err=1e-5)
      self.assertNear(lr_schedule(1), 2.38298470417, err=1e-5)
      self.assertNear(lr_schedule(10), 0.3, err=1e-5)
      self.assertNear(lr_schedule(25), 0.00948683298, err=1e-5)

    with flag_sandbox({f'{TEST_CLIENT_FLAG_PREFIX}_learning_rate': 3.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_schedule': 'exp_decay', f'{TEST_CLIENT_FLAG_PREFIX}_lr_warmup_steps': 10, f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_steps': 10, f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_rate': 0.1, f'{TEST_CLIENT_FLAG_PREFIX}_lr_staircase': False}):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 0.3, err=1e-5)
      self.assertNear(lr_schedule(1), 0.6, err=1e-5)
      self.assertNear(lr_schedule(10), 3.0, err=1e-5)
      self.assertNear(lr_schedule(11), 2.38298470417, err=1e-5)
      self.assertNear(lr_schedule(20), 0.3, err=1e-5)
      self.assertNear(lr_schedule(35), 0.00948683298, err=1e-5)

  def test_create_inv_lin_client_lr_schedule_from_flags(self):
    with flag_sandbox({f'{TEST_CLIENT_FLAG_PREFIX}_learning_rate': 5.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_schedule': 'inv_lin_decay', f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_steps': 10, f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_rate': 10.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_staircase': True}):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 5.0, err=1e-5)
      self.assertNear(lr_schedule(1), 5.0, err=1e-5)
      self.assertNear(lr_schedule(10), 0.454545454545, err=1e-5)
      self.assertNear(lr_schedule(19), 0.454545454545, err=1e-5)
      self.assertNear(lr_schedule(20), 0.238095238095, err=1e-5)

    with flag_sandbox({f'{TEST_CLIENT_FLAG_PREFIX}_learning_rate': 5.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_schedule': 'inv_lin_decay', f'{TEST_CLIENT_FLAG_PREFIX}_lr_warmup_steps': 0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_steps': 10, f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_rate': 10.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_staircase': False}):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 5.0, err=1e-5)
      self.assertNear(lr_schedule(1), 2.5, err=1e-5)
      self.assertNear(lr_schedule(9), 0.5, err=1e-5)
      self.assertNear(lr_schedule(19), 0.25, err=1e-5)

    with flag_sandbox({f'{TEST_CLIENT_FLAG_PREFIX}_learning_rate': 5.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_schedule': 'inv_lin_decay', f'{TEST_CLIENT_FLAG_PREFIX}_lr_warmup_steps': 10, f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_steps': 10, f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_rate': 10.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_staircase': False}):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 0.5, err=1e-5)
      self.assertNear(lr_schedule(1), 1.0, err=1e-5)
      self.assertNear(lr_schedule(10), 5.0, err=1e-5)
      self.assertNear(lr_schedule(11), 2.5, err=1e-5)
      self.assertNear(lr_schedule(19), 0.5, err=1e-5)
      self.assertNear(lr_schedule(29), 0.25, err=1e-5)

  def test_create_inv_sqrt_client_lr_schedule_from_flags(self):
    with flag_sandbox({f'{TEST_CLIENT_FLAG_PREFIX}_learning_rate': 2.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_schedule': 'inv_sqrt_decay', f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_steps': 10, f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_rate': 10.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_staircase': True}):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 2.0, err=1e-5)
      self.assertNear(lr_schedule(1), 2.0, err=1e-5)
      self.assertNear(lr_schedule(10), 0.603022689155, err=1e-5)
      self.assertNear(lr_schedule(19), 0.603022689155, err=1e-5)
      self.assertNear(lr_schedule(20), 0.436435780472, err=1e-5)

    with flag_sandbox({f'{TEST_CLIENT_FLAG_PREFIX}_learning_rate': 2.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_schedule': 'inv_sqrt_decay', f'{TEST_CLIENT_FLAG_PREFIX}_lr_warmup_steps': 0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_steps': 10, f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_rate': 10.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_staircase': False}):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 2.0, err=1e-5)
      self.assertNear(lr_schedule(3), 1.0, err=1e-5)
      self.assertNear(lr_schedule(99), 0.2, err=1e-5)
      self.assertNear(lr_schedule(399), 0.1, err=1e-5)

    with flag_sandbox({f'{TEST_CLIENT_FLAG_PREFIX}_learning_rate': 2.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_schedule': 'inv_sqrt_decay', f'{TEST_CLIENT_FLAG_PREFIX}_lr_warmup_steps': 10, f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_steps': 10, f'{TEST_CLIENT_FLAG_PREFIX}_lr_decay_rate': 10.0, f'{TEST_CLIENT_FLAG_PREFIX}_lr_staircase': False}):
      lr_schedule = optimizer_utils.create_lr_schedule_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertNear(lr_schedule(0), 0.2, err=1e-5)
      self.assertNear(lr_schedule(1), 0.4, err=1e-5)
      self.assertNear(lr_schedule(10), 2.0, err=1e-5)
      self.assertNear(lr_schedule(13), 1.0, err=1e-5)
      self.assertNear(lr_schedule(109), 0.2, err=1e-5)
      self.assertNear(lr_schedule(409), 0.1, err=1e-5)


class RemoveUnusedFlagsTest(tf.test.TestCase):

  def test_remove_unused_flags_without_optimizer_flag(self):
    hparam_dict = collections.OrderedDict([('client_opt_fn', 'sgd'),
                                           ('client_sgd_momentum', 0.3)])
    with self.assertRaisesRegex(ValueError,
                                'The flag client_optimizer was not defined.'):
      _ = optimizer_utils.remove_unused_flags('client', hparam_dict)

  def test_remove_unused_flags_with_empty_optimizer(self):
    hparam_dict = collections.OrderedDict([('optimizer', '')])

    with self.assertRaisesRegex(
        ValueError, 'The flag optimizer was not set. '
        'Unable to determine the relevant optimizer.'):
      _ = optimizer_utils.remove_unused_flags(
          prefix=None, hparam_dict=hparam_dict)

  def test_remove_unused_flags_with_prefix(self):
    hparam_dict = collections.OrderedDict([('client_optimizer', 'sgd'),
                                           ('non_client_value', 0.1),
                                           ('client_sgd_momentum', 0.3),
                                           ('client_adam_momentum', 0.5)])

    relevant_hparam_dict = optimizer_utils.remove_unused_flags(
        'client', hparam_dict)
    expected_flag_names = [
        'client_optimizer', 'non_client_value', 'client_sgd_momentum'
    ]
    self.assertCountEqual(relevant_hparam_dict.keys(), expected_flag_names)
    self.assertEqual(relevant_hparam_dict['client_optimizer'], 'sgd')
    self.assertEqual(relevant_hparam_dict['non_client_value'], 0.1)
    self.assertEqual(relevant_hparam_dict['client_sgd_momentum'], 0.3)

  def test_remove_unused_flags_without_prefix(self):
    hparam_dict = collections.OrderedDict([('optimizer', 'sgd'), ('value', 0.1),
                                           ('sgd_momentum', 0.3),
                                           ('adam_momentum', 0.5)])
    relevant_hparam_dict = optimizer_utils.remove_unused_flags(
        prefix=None, hparam_dict=hparam_dict)
    expected_flag_names = ['optimizer', 'value', 'sgd_momentum']
    self.assertCountEqual(relevant_hparam_dict.keys(), expected_flag_names)
    self.assertEqual(relevant_hparam_dict['optimizer'], 'sgd')
    self.assertEqual(relevant_hparam_dict['value'], 0.1)
    self.assertEqual(relevant_hparam_dict['sgd_momentum'], 0.3)

  def test_removal_with_standard_default_values(self):
    hparam_dict = collections.OrderedDict([('client_optimizer', 'adam'),
                                           ('non_client_value', 0),
                                           ('client_sgd_momentum', 0),
                                           ('client_adam_param1', None),
                                           ('client_adam_param2', False)])

    relevant_hparam_dict = optimizer_utils.remove_unused_flags(
        'client', hparam_dict)
    expected_flag_names = [
        'client_optimizer', 'non_client_value', 'client_adam_param1',
        'client_adam_param2'
    ]
    self.assertCountEqual(relevant_hparam_dict.keys(), expected_flag_names)
    self.assertEqual(relevant_hparam_dict['client_optimizer'], 'adam')
    self.assertEqual(relevant_hparam_dict['non_client_value'], 0)
    self.assertIsNone(relevant_hparam_dict['client_adam_param1'])
    self.assertEqual(relevant_hparam_dict['client_adam_param2'], False)

  def test_remove_flags_with_optimizers_sharing_a_prefix(self):
    hparam_dict = collections.OrderedDict([('client_optimizer', 'adamW'),
                                           ('client_adam_momentum', 0.3),
                                           ('client_adamW_momentum', 0.5)])
    relevant_hparam_dict = optimizer_utils.remove_unused_flags(
        'client', hparam_dict)
    expected_flag_names = ['client_optimizer', 'client_adamW_momentum']
    self.assertCountEqual(relevant_hparam_dict.keys(), expected_flag_names)
    self.assertEqual(relevant_hparam_dict['client_optimizer'], 'adamW')
    self.assertEqual(relevant_hparam_dict['client_adamW_momentum'], 0.5)


if __name__ == '__main__':
  tf.test.main()
