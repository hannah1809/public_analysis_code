#!/usr/bin/env python

from setuptools import setup

setup(name='ibc_public',
      version='0.1',
      description='Public code for IBC data analysis',
      url='https://github.com/hbp-brain-charting/public_analysis_code',
      author='Bertrand Thirion',
      author_email='bertrand.thirion@inria.fr',
      packages=['ibc_public'],
      data_files=[('masks', ['ibc_data/gm_mask_1_5mm.nii.gz', 
                             'ibc_data/gm_mask_3mm.nii.gz'])],
)
