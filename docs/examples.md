# Examples

This page provides example usages of the `enc4ppm` package.

## Basic example

The following example shows how to setup encoder to properly read log columns like case id, activity name, etc.

```python
import pandas as pd

from enc4ppm.frequency_encoder import FrequencyEncoder
from enc4ppm.constants import LabelingType

log = pd.read_csv('log.csv')

encoder = FrequencyEncoder(
    labeling_type=LabelingType.NEXT_ACTIVITY,
    case_id_key='CaseID',               # if not set, defaults to case:concept:name
    activity_key='Activity',            # if not set, defaults to concept:name
    timestamp_key='Complete Timestamp', # if not set, defaults to time:timestamp
)

encoded_log = encoder.encode(log)
```

## Encode train-test data: freezing the encoder

When working with train-test data, we don't want the encoder to look at test data to avoid data leakage. It is usually a good idea to first encode training data and freeze the encoder on it, then use the frozen encoder to encode test data.

Encoder can be frozen by calling `.encode()` with `freeze=True`. When an encoder is not frozen, calling `.encode()`  will build internal vocabularies of activities, attributes, and so on; on the other hand, if `.encode()` is called on a frozen encoder it will use the previously computed internal vocabularies.

For example, test set may contain an activity `ActivityX` that is not present in training set. If frequency encoding is performed on the full log (training+test), then the encoded dataframe will contain a column `ActivityX` even though that activity should not be known during training.

The following example avoids this problem by encoding training and test sets separately, freezing the encoder on the training data.

```python
import pandas as pd

from enc4ppm.simple_index_encoder import SimpleIndexEncoder
from enc4ppm.constants import LabelingType, PrefixStrategy

def split_log(log):
    # your split logic here

log = pd.read_csv('log.csv')

train_log, test_log = split_log(log)                       # split the log before encoding

encoder = SimpleIndexEncoder(
    labeling_type=LabelingType.REMAINING_TIME
)

encoded_train_log = encoder.encode(train_log, freeze=True) # freeze encoder on train log
encoded_test_log = encoder.encode(test_log)                # use frozen encoder on test log
```

The encoder is frozen on the train log. As a result, when encoding the test log, `ActivityX` will be mapped to activity `UNKNOWN`, because it was not present in train log.

## Save and load encoder to disk

You can save the encoder object to a file and load it for later use. In order to save an encoder to disk you need to freeze it first.

The following example shows a simple save/load worflow.

```python
import pandas as pd

from enc4ppm.frequency_encoder import FrequencyEncoder
from enc4ppm.constants import LabelingType

log = pd.read_csv('log.csv')

encoder = FrequencyEncoder(
    labeling_type=LabelingType.NEXT_ACTIVITY
)

encoded_log = encoder.encode(log, freeze=True)              # freeze encoder
encoder.summary()                                           # print info about the encoder

encoder.save('/path/to/encoder.pkl')                        # save encoder to disk

loaded_encoder = BaseEncoder.load('/path/to/encoder.pkl')   # load encoder from disk
loaded_encoder.summary()                                    # print info about loaded_encoder (should output the same as encoder.summary())

inference_log = pd.read_csv('inference.csv')
encoded_inference_log = loaded_encoder.encode(inference_log)
```

## Prefix length and strategy

You can specify `prefix_length` to set a specific prefix length, otherwise the maximum prefix length found in the log will be used. You can specify `prefix_strategy` to be either `up_to_specified` (the default) which will consider all prefix lengths from 1 up to `prefix_length`, or `only_specified` which will consider only prefix of length `prefix_length`.

The following code encodes a log keeping only examples with a prefix length of 10.

```python
import pandas as pd

from enc4ppm.simple_index_encoder import SimpleIndexEncoder
from enc4ppm.constants import LabelingType, PrefixStrategy

log = pd.read_csv('log.csv')

encoder = SimpleIndexEncoder(
    labeling_type=LabelingType.REMAINING_TIME,
    prefix_length=10,
    prefix_strategy=PrefixStrategy.ONLY_SPECIFIED,
)

encoded_log = encoder.encode(log)
```

## Categorical encoding

The `categorical_encoding` parameter determines whether categorical values (activity names and categorical attributes) are kept as `string` (default) or `one-hot` encoded.

The following example encodes with one-hot encoding.

```python
import pandas as pd

from enc4ppm.simple_index_encoder import SimpleIndexEncoder
from enc4ppm.constants import LabelingType, CategoricalEncoding

log = pd.read_csv('log.csv')

encoder = SimpleIndexEncoder(
    labeling_type=LabelingType.REMAINING_TIME,
    categorical_encoding=CategoricalEncoding.ONE_HOT,
)

encoded_log = encoder.encode(log)
```

## Numerical scaling

The `numerical_scaling` parameter can be used to scale numerical values (numerical attributes, label in the case of remaining time, and TimeSinceCaseStart and TimeSincePreviousActivity features). It can be either `none` (default) to not apply any scaling, or `standardization` to apply standardization. The dictionary `encoder.numerical_scaling_info` will contain `mean` and `std` values to transform standardized numerical values back to their original range. `unscale_numerical_feature` is a helper method that unscales standardization automatically.

The following example first standardizes all numerical values, then restores the 'label' column back to its original space.

```python
import pandas as pd

from enc4ppm.frequency_encoder import FrequencyEncoder
from enc4ppm.constants import LabelingType, NumericalScaling

log = pd.read_csv('log.csv')

encoder = FrequencyEncoder(
    labeling_type=LabelingType.REMAINING_TIME,
    numerical_scaling=NumericalScaling.STANDARDIZATION,
)

encoded_log = encoder.encode(log)

# Use encoder.numerical_scaling_info to restore original label values...
restored_label = encoder.numerical_scaling_info[encoder.LABEL_KEY]['std'] * encoded_log[encoder.LABEL_KEY] + encoder.numerical_scaling_info[encoder.LABEL_KEY]['mean']

# ... or use the method unscale_numerical_feature()
restored_label = encoder.unscale_numerical_feature(encoded_log[encoder.LABEL_KEY], encoder.LABEL_KEY)

```

## Label remaining time as a classification task

Instead of labeling remaining time as a regression task (with label being the number of hours for remaining trace completion), it is also possible to label it as a classification task.

The following example labels remaining time as a classification problem, also specifying the number of bins to divide times in.

```python
import pandas as pd

from enc4ppm.simple_index_encoder import SimpleIndexEncoder
from enc4ppm.constants import LabelingType

log = pd.read_csv('log.csv')

encoder = SimpleIndexEncoder(
    labeling_type=LabelingType.REMAINING_TIME_CLASSIFICATION
)
encoder.set_remaining_time_num_bins(10) # cut in 10 bins (10 classes)

encoded_log = encoder.encode(log)
```
