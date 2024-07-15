from mipha.factory import make_feature_extractor


def test_make_feature_extractor():
    data = [0, 1, 2, 3]
    test_extract = lambda test_list: len(test_list)

    TestFE = make_feature_extractor(test_extract)

    assert TestFE().extract_features(data) == 4

