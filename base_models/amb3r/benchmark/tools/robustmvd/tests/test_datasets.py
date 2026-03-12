# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/tests/test_datasets.py

import rmvd.data

def test_build_dataset_names():
    assert rmvd.data.registry._build_dataset_name('eth3d.robustmvd.mvd') == 'eth3d.robustmvd.mvd'
    assert rmvd.data.registry._build_dataset_name('eth3d.robustmvd.mvd', dataset_type='mvd') == 'eth3d.robustmvd.mvd'
    assert rmvd.data.registry._build_dataset_name('eth3d.robustmvd.mvd', split='robustmvd') == 'eth3d.robustmvd.mvd'
    assert rmvd.data.registry._build_dataset_name('eth3d.robustmvd.mvd', dataset_type='mvd', split='robustmvd') == 'eth3d.robustmvd.mvd'
    assert rmvd.data.registry._build_dataset_name('eth3d.robustmvd', dataset_type='mvd') == 'eth3d.robustmvd.mvd'
    assert rmvd.data.registry._build_dataset_name('eth3d.mvd', split='robustmvd') == 'eth3d.robustmvd.mvd'
    assert rmvd.data.registry._build_dataset_name('eth3d.mvd') == 'eth3d.robustmvd.mvd'
    assert rmvd.data.registry._build_dataset_name('eth3d', dataset_type='mvd', split='robustmvd') == 'eth3d.robustmvd.mvd'
    assert rmvd.data.registry._build_dataset_name('eth3d.robustmvd') == 'eth3d.robustmvd'
    assert rmvd.data.registry._build_dataset_name('eth3d') == 'eth3d'


def test_split_dataset_names():
    assert rmvd.data.registry._split_dataset_name("eth3d.robustmvd.mvd", None, None) == ("eth3d", "mvd", "robustmvd")
    assert rmvd.data.registry._split_dataset_name("eth3d.robustmvd", "mvd", None) == ("eth3d", "mvd", "robustmvd")
    assert rmvd.data.registry._split_dataset_name("eth3d", "mvd", "robustmvd") == ("eth3d", "mvd", "robustmvd")
    assert rmvd.data.registry._split_dataset_name("eth3d.robustmvd", "mvd", "robustmvd") == ("eth3d", "mvd", "robustmvd")
    assert rmvd.data.registry._split_dataset_name("eth3d.mvd", "mvd", "robustmvd") == ("eth3d", "mvd", "robustmvd")
    assert rmvd.data.registry._split_dataset_name("eth3d.mvd", None, "robustmvd") == ("eth3d", "mvd", "robustmvd")
    assert rmvd.data.registry._split_dataset_name("eth3d.mvd", None, None) == ("eth3d", "mvd", "robustmvd")
    assert rmvd.data.registry._split_dataset_name("eth3d", "mvd", None) == ("eth3d", "mvd", "robustmvd")
