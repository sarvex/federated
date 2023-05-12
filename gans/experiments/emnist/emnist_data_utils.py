# Copyright 2019, Google LLC.
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
"""Data utilities for working with Federated EMNIST dataset."""

import functools

import tensorflow as tf
import tensorflow_federated as tff

from utils.datasets import infinite_emnist

BATCH_SIZE = 32
SHUFFLE_BUFFER = 10000


def _get_transformed_client_data(
    client_data: tff.simulation.datasets.ClientData, num_pseudo_clients: int):
  """Returns a (potentially) transformed version of EMNIST client data."""
  if num_pseudo_clients == 1:
    return client_data
  else:
    return infinite_emnist.get_infinite(
        client_data, client_expansion_factor=num_pseudo_clients)


@functools.lru_cache(maxsize=1)
def create_real_images_tff_client_data(split='train', num_pseudo_clients=1):
  """Returns `tff.simulation.datasets.ClientData` of real images of numbers/letters."""

  if split == 'synthetic':
    client_data = tff.simulation.datasets.emnist.get_synthetic()
  else:
    train_tff_data, eval_tff_data = tff.simulation.datasets.emnist.load_data(
        only_digits=False)
    if split == 'train':
      client_data = train_tff_data
    elif split == 'test':
      client_data = eval_tff_data

  if split in ['synthetic', 'train', 'test']:
    return _get_transformed_client_data(client_data, num_pseudo_clients)
  elif split == 'both':
    return (_get_transformed_client_data(train_tff_data, num_pseudo_clients),
            _get_transformed_client_data(eval_tff_data, num_pseudo_clients))
  else:
    raise ValueError(f'Unknown dataset split {split}')


def preprocess_img_dataset(images_ds,
                           invert_imagery=False,
                           include_label=False,
                           batch_size=BATCH_SIZE,
                           shuffle=True,
                           repeat=False):
  """Returns a preprocessed dataset.

  The preprocessing converts the raw image from [0.0, 1.0] (where 1.0 is
  background) to [-1.0, 1.0] (where -1.0 is background). The range change is in
  order to put the images in a format that's amenable to GAN discriminator
  input. If specified, the image pixel intensities will be inverted (i.e., the
  background will be changed to 1.0).

  The preprocessing also converts the raw label from [0, 61] (where [0, 9] are
  labels for numbers, [10, 35] are labels for uppercase letters, and [36, 61]
  are labels for lowercase letters) to [0, 35] (where [0, 9] are labels for
  numbers and [10, 35] are labels for letters). In other words, the labels are
  converted to be letter case agnostic.

  Args:
    images_ds: The raw EMNIST dataset of OrderedDict elements, to be processed.
    invert_imagery: If True, invert the pixel intensities of the images.
    include_label: If False, the output dataset is only images. If True, the
      output dataset includes labels along with images.
    batch_size: Batch size of output dataset. If None, don't batch.
    shuffle: If True, shuffle the dataset.
    repeat: If True, repeat the dataset.

  Returns:
    A preprocessed, batched, and possibly shuffled/repeated dataset of images
    (and possibly labels).
  """

  @tf.function
  def _preprocess(element):
    """Preprocess: invert image if specified, and make label case agnostic."""
    image = tf.expand_dims(element['pixels'], 2)
    image = -2.0 * (image - 0.5)
    if invert_imagery:
      image = -1.0 * image

    if include_label:
      # Reduce label set to be [0, 35], where [0, 9] correspond to numbers and
      # [10, 35] correspond to letters. This makes our label set case agnostic.
      label = element['label']
      if label >= 36:
        label -= 26
      return image, label

    return image

  images_ds = images_ds.map(_preprocess)

  if shuffle:
    images_ds = images_ds.shuffle(
        buffer_size=SHUFFLE_BUFFER, reshuffle_each_iteration=True, seed=124578)
  # Shuffle comes before repeat so we shuffle within each epoch,
  # but process complete epochs before repeating.
  if repeat:
    images_ds = images_ds.repeat()

  if batch_size is not None:
    images_ds = images_ds.batch(batch_size, drop_remainder=False)
  return images_ds.prefetch(tf.data.experimental.AUTOTUNE)
