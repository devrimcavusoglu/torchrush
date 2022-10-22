import json


def json_load(fp: str, **kwargs):
    with open(fp, "r") as jf:
        return json.load(jf, **kwargs)


def json_save(obj: object, fp: str, **kwargs):
    with open(fp, "w") as jf:
        json.dump(obj, jf, **kwargs)
