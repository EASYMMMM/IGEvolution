"""Equivalence to the original Conv1d TCN, including old optimizer checkpoints."""
import copy
import io
from types import MethodType
import unittest

import isaacgym  # noqa: F401 -- must precede torch
import torch
from torch.nn import functional as F

from isaacgymenvs.learning.SRLEvo.hmi_encoder.config import EncoderConfig
from isaacgymenvs.learning.SRLEvo.hmi_encoder.model import CausalBlock, HMIEncoder


def conv_reference(self, x):
    """Original implementation; retained only for tests and benchmarks."""
    y = self.conv(F.pad(x, (self.dilation, 0)))
    y = self.norm(y.transpose(1, 2)).transpose(1, 2)
    return x + F.elu(y)


def as_conv_reference(model):
    for block in model.modules():
        if isinstance(block, CausalBlock):
            block.forward = MethodType(conv_reference, block)
    return model


class TCNEquivalenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)

    def test_block_outputs_input_and_all_parameter_gradients(self):
        for length, dilation in ((64, 1), (64, 2), (64, 4), (64, 8),
                                 (64, 16), (64, 32), (3, 3), (3, 8)):
            with self.subTest(length=length, dilation=dilation):
                fast = CausalBlock(5, dilation).double()
                old = as_conv_reference(copy.deepcopy(fast))
                # Match the noncontiguous channel-first layout used by HMIEncoder.
                x = torch.randn(2, length, 5, dtype=torch.float64).transpose(1, 2)
                x_fast = x.detach().requires_grad_(True)
                x_old = x.clone().detach().requires_grad_(True)
                actual, expected = fast(x_fast), old(x_old)
                torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
                upstream = torch.randn_like(actual)
                actual.backward(upstream)
                expected.backward(upstream)
                torch.testing.assert_close(x_fast.grad, x_old.grad, rtol=1e-10, atol=1e-11)
                for (name, p), (old_name, q) in zip(fast.named_parameters(), old.named_parameters()):
                    self.assertEqual(name, old_name)
                    torch.testing.assert_close(p.grad, q.grad, rtol=1e-10, atol=1e-11)

    def test_full_encoder_output_and_gradients_float32(self):
        fast = HMIEncoder(EncoderConfig(enabled=True))
        old = as_conv_reference(copy.deepcopy(fast))
        x = torch.randn(3, 64, 41)
        x[..., -1] = 1
        x[0, :23] = 0  # partial history after reset
        a = x.clone().requires_grad_(True)
        b = x.clone().requires_grad_(True)
        z, prediction = fast(a)
        old_z, old_prediction = old(b)
        torch.testing.assert_close(z, old_z, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(prediction, old_prediction, rtol=2e-5, atol=2e-6)
        loss_z, loss_prediction = torch.randn_like(z), torch.randn_like(prediction)
        (z.mul(loss_z).sum() + prediction.mul(loss_prediction).sum()).backward()
        (old_z.mul(loss_z).sum() + old_prediction.mul(loss_prediction).sum()).backward()
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=1e-5)
        for name, p in fast.named_parameters():
            torch.testing.assert_close(p.grad, dict(old.named_parameters())[name].grad,
                                       rtol=2e-4, atol=1e-5, msg=name)

    def test_legacy_model_and_adam_checkpoint_next_step(self):
        old = as_conv_reference(HMIEncoder(EncoderConfig(enabled=True)).double())
        old_opt = torch.optim.Adam(old.parameters(), lr=3e-4)
        x = torch.randn(2, 64, 41, dtype=torch.float64)
        x[..., -1] = 1

        def step(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            z, pred = model(x)
            (z.square().mean() + pred.square().mean()).backward()
            optimizer.step()

        step(old, old_opt)
        buffer = io.BytesIO()
        torch.save({'model': old.state_dict(), 'optimizer': old_opt.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, map_location='cpu')
        fast = HMIEncoder(EncoderConfig(enabled=True)).double()
        fast.load_state_dict(checkpoint['model'], strict=True)
        fast_opt = torch.optim.Adam(fast.parameters(), lr=3e-4)
        fast_opt.load_state_dict(checkpoint['optimizer'])
        self.assertEqual(list(old.state_dict()), list(fast.state_dict()))
        self.assertEqual([n for n, _ in old.named_parameters()],
                         [n for n, _ in fast.named_parameters()])
        step(old, old_opt)
        step(fast, fast_opt)
        for name, value in fast.state_dict().items():
            torch.testing.assert_close(value, old.state_dict()[name], rtol=1e-9, atol=1e-11)
        # The resulting checkpoint can also be read by the original implementation.
        old.load_state_dict(fast.state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()
