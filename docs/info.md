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

I lobotomized the makefile and test.py to integrate my own custom pytest cocotb flow. To setup the environment first run:

```bash
pip3 install -r test/requirements.txt
```

Then open the tests directory and run the cocotb pytest:

```bash
cd tests
make -B
```

You can run specific tests by attaching a keyword:

```bash
make -b ARGS="reset" // Runs all reset tests
make -b ARGS="width" // Runs all tests in width category
make -b ARGS="32"    // Runs all tests with a parameter of 32
```

The Icarus simulator is leveraged to test two parameterized categories: width and channels.

- reset test checks connectivity
- single test checks single data input to output

These parameterized runs are then tested over varying rates of readiness for 100 inputs.
- in_fuzz where input is ready 50%
- out_fuzz where output is ready 50%
- inout_fuzz where both are ready 50%
- full_bw where both are ready 100%.

## External hardware

No physical testing yet, but will be implemented on an Icebreaker v1.1a FPGA for final project. Github Actions respects my pytest workflow, but my ANN was not designed with GDS pincount in mind and will not pass the GDS action. This is accounted for in my final project where the result is put back onto a single pin via bit-packer and framing hardware.
