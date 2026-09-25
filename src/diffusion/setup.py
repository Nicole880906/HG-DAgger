from setuptools import setup, find_namespace_packages

setup(
  name = 'diffusion_policy',
  version = '0.1.0',
  packages = find_namespace_packages(include=['diffusion_policy', 'diffusion_policy.*']),
)
