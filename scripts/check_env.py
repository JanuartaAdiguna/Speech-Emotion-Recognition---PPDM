import importlib
packages = ['torch', 'librosa', 'soundfile', 'numpy']
for p in packages:
    try:
        m = __import__(p)
        ver = getattr(m, '__version__', 'unknown')
        print(f'{p}: INSTALLED ({ver})')
    except Exception as e:
        print(f'{p}: NOT INSTALLED ({e.__class__.__name__})')
