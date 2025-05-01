# save

## Function

```python
crisgi_obj.save()
```

This function serializes the CRISGI object and saves it as a pickle file `*.pk` in the output directory, using the dataset name as part of the filename.

## Parameters

| Name      | Type   | Description                                      |
|-----------|--------|--------------------------------------------------|
| self      | object | The instance of the CRISGI class to be saved.    |

## Return type

`None`

## Returns

This function does not return any value. It performs the side effect of saving the object to a file.

## Attributes Set

- `out_dir`: The directory where the pickle file will be saved.
- `dataset`: The name of the dataset, used in the filename.

## Example

```python
# Assume crisgi_obj is an instance of the CRISGI class
crisgi_obj.out_dir = '/path/to/output'
crisgi_obj.dataset = 'my_dataset'

# Save the CRISGI object to a pickle file
crisgi_obj.save()

# Output:
# [Output] CRISGI object has benn saved to:
# /path/to/output/my_dataset_crisgi_obj.pk
```

This example demonstrates how to set the output directory and dataset name before calling `save()` to persist the CRISGI object.
