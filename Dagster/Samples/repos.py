import sys

from dagster import repository
from dagster.utils import script_relative_path

sys.path.append(script_relative_path("."))

from hello_cereal import hello_cereal_pipeline
from hello_dagster import hello_pipeline

@repository
def hello_cereal_repository():
    # Note that we can pass a dict of functions, rather than a list of
    # pipeline definitions. This allows us to construct pipelines lazily,
    # if, e.g., initializing a pipeline involves any heavy compute
    return {
        "pipelines": {
            "hello_cereal_pipeline": lambda: hello_cereal_pipeline,
            "hello_pipeline":  lambda: hello_pipeline
        }
    }