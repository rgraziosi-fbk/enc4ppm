# Encoding For Predictive Process Monitoring (enc4ppm)

`enc4ppm` is a Python package than provides common process mining encodings.

## Installation

Using pip:

`pip install enc4ppm`

## Example

The following example performs frequency encoding with latest payload for next activity prediction task:

```python
import pandas as pd

from enc4ppm.frequency_encoder import FrequencyEncoder
from enc4ppm.constants import LabelingType

# Load log
log = pd.read_csv('bpic2012.csv')

# Create encoder
encoder = FrequencyEncoder(
    labeling_type=LabelingType.NEXT_ACTIVITY,
    include_latest_payload=True,
    attributes=['AMOUNT_REQ'],
)

# Encode log
encoded_log = encoder.encode(log)
```