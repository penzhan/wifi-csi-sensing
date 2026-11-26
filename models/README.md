# Pine inference assets

- `pine_shared_lenet_concat.pt`: SharedChainPineLeNet checkpoint trained for
  300 epochs with concat fusion. Input shape: `4 × 500 × 128`; recorded test
  accuracy: `61.43%`.

The browser never embeds a prediction timeline. After the simulated capture
collects 500 frames, it calls `serve_pine_inference.py`; the server runs the
`.pt` checkpoint and returns the label and Softmax confidence at that moment.
