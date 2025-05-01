# detect_startpoint

## Function

```python
crisgi_obj.detect_startpoint(symptom_types = ["Symptomatic"])
```

Detects the start point (CT_time) for samples with specified symptom types and updates the `CT_time` column in `edata.obs`. This method filters samples based on the provided symptom types, processes each subject's data, applies a start point detection algorithm, and stores the predicted start time for each subject.

## Parameters

| Name           | Type         | Description                                                                                 |
|----------------|--------------|--------------------------------------------------------------------------------------------|
| symptom_types  | list of str  | A list of symptom types to filter samples by. Should match values in `edata.obs['symptom']`. Default is `["Symptomatic"]`. |

## Return type

None

## Returns

This function does not return a value. It updates the `CT_time` attribute in the `edata.obs` DataFrame for each subject matching the specified symptom types.

## Attributes Set

- **edata.obs['CT_time']**: The predicted start time (CT_time) for each subject with the specified symptom types.

## Example

```python
# Assume CRISGI is already imported and instantiated as crisgi
# and crisgi.edata is properly initialized

# Detect start points for symptomatic subjects (default)
crisgi.detect_startpoint()

# Detect start points for both symptomatic and asymptomatic subjects
crisgi.detect_startpoint(symptom_types=["Symptomatic", "Asymptomatic"])

# After running, check the predicted CT_time for each subject
print(crisgi.edata.obs[['subject', 'CT_time']].drop_duplicates())
```
