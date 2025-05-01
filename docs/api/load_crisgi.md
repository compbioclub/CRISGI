# load_crisgi

## Function
```python
load_crisgi(pk_fn)
```

Loads a CRISGI object from a specified pickle file and prints a confirmation message upon successful loading.

## Parameters

| Name   | Type   | Description                                      |
|--------|--------|--------------------------------------------------|
| pk_fn  | str    | Path to the pickle file containing the CRISGI object. |

## Return type

`object`

## Returns

The CRISGI object that was loaded from the specified pickle file.

## Attributes Set

This function does not set any attributes on the returned object or elsewhere.

## Example

```python
import pickle

def print_msg(msg):
    print(msg)

def load_crisgi(pk_fn):
    crisgi_obj = pickle.load(open(pk_fn, 'rb'))
    print_msg(f'[Input] CRISGI object stored at {pk_fn} has been loaded.')
    return crisgi_obj

# Example usage:
# Assume you have previously saved a CRISGI object to 'crisgi_obj.pkl'
# To load it:

crisgi = load_crisgi('crisgi_obj.pkl')
# Output: [Input] CRISGI object stored at crisgi_obj.pkl has been loaded.
```
