# VAE neuron-sketch synthesis

Generate neuron-feature samples and render HDF5 sketches with ROI JSON annotations.

```bash
python -m pip install -r requirements.txt
python generate_neuron_sketches.py --help
python generate_neuron_sketches.py --data-file /path/to/features.xlsx --output-dir /path/to/sketches
python scripts/postprocess_corpus.py /path/to/sketches
```

The other generation entry points provide Gaussian sampling, empirical-marginal sampling, configurable VAE models, and spherical shapes. Postprocessing writes TIFF volumes and center JSON in ZYX order.

[MIT license](LICENSE) · [Citation](CITATION.cff)
