<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

My project implements a parameterizeable linear layer functionally identical to Pytorch.linear().
In an Artificial Neural Network (ANN) capable of deep learning, several layers with increasing
amounts of features are connected in series in order to learn more abstract associations and features.

My layer includes parameters defining data widths for input and output, as well as weight and bias, which
play a large role in Quantization-Aware Training (QAT), my goal for the ECE210 final project.

These weights and biases are cooked into the neuron's architecture and so are parameters defined once at synthesis
rather than dynamic inputs which change cycle-to-cycle. Due to python cocotb's restrictions on data type width,
bit-vectors longer than 32 bits cannot be passed in via the CLI. Therefore they are generated into .vh files
and exposed via python's environment variables.

The other parameters define the shape of the linear layer:
```python
    fc = torch.nn.Linear(in_features=InChannels, out_features=OutChannels, bias=True)
```
OutChannels determines the number of outputs of the layer. Each neuron produces its own output, therefore
we generate this amount of neurons. 

InChannels defines how many parallel data inputs the layer receives, and being fully-connected, every neuron receives every input. Each neuron imports a weight for every input, so subsequently each neuron must be parameterized with this many weights. These weights are multiplied with every input and then summed and added
to a bias value.

## How to test

I lobotomized the makefile and test.py to integrate my own custom pytest cocotb flow. Running "make -B" Should run icarus tests for my two parameterized categories test_width, and test_channels. These parameterized runs are then tested over varying rates of readyness. Infuzz where input is ready 50%, Outfuzz where output is ready 50%, Inoutfuzz where both are ready 50%, and fullbw where both are ready 100%.

## External hardware

No physical testing yet, but will be implemented on an Icebreaker v1.1a FPGA for final project.
