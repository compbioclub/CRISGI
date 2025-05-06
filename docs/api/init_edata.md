# init_edata

## Function

```python
crisgi_obj.init_edata(test_obss, headers)
```

Initializes the `edata` attribute for the object by extracting and organizing observation and interaction data from the underlying AnnData structure. This function prepares a new AnnData object (`edata`) containing selected observations and gene-gene interaction information, and sets it as an attribute for downstream analysis.

## Parameters

| Name      | Type            | Description                                                                 |
|-----------|-----------------|-----------------------------------------------------------------------------|
| test_obss  | list of list    | A list where each element is a list of observation indices or labels to include (test populations). |
| headers   | list of str     | List of column names from the observation metadata to retain in `edata.obs`.   |

## Return type

None

## Returns

This function does not return a value. It sets the `edata` attribute on the object with a new AnnData instance containing the selected observations and interaction data.

## Attributes Set

- **self.edata**: `AnnData`  
  The newly created AnnData object containing the selected observations and gene-gene interaction information.

## Example

```python
# Suppose `crisgi_obj` is an instance of the class containing `init_edata`
# test_obss is a list of lists of observation indices or labels (test populations)
test_obss = [
        ['cell_1', 'cell_2'],
        ['cell_3', 'cell_4']
]
# headers are the columns to keep from the observation metadata
headers = ['test', 'group']

crisgi_obj.init_edata(test_obss, headers)

# After calling, crisgi_obj.edata will be available for further analysis:
print(crisgi_obj.edata.shape)  # (number of observations, number of interactions)
print(crisgi_obj.edata.obs.head())  # Shows the selected observation metadata
print(crisgi_obj.edata.var.head())  # Shows gene interaction information
```
