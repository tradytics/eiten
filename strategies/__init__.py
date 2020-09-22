from os.path import dirname, basename, isfile, join
import glob
import importlib.util
import inspect

modules = glob.glob(join(dirname(__file__), "*.py"))
__names = [basename(f)[:-3] for f in modules if isfile(f)
           and not f.endswith('__init__.py')]
portfolios = []
for i in __names:
    spec = importlib.util.spec_from_file_location("", f"./strategies/{i}.py")
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    portfolios.append(inspect.getmembers(foo, inspect.isclass)[0][1]())
