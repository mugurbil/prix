#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the TCP transport and its pickle-free wire format."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import prix
from prix.communicator import wire


def _workload():
    x = prix.array(np.linspace(-3, 3, 64).astype(np.float32))
    w = prix.array(np.full((64, 8), 1 / 64, np.float32))
    big = prix.array(np.ones((1500, 1500), np.float32))  # larger than one shared-memory slot
    return (
        np.asarray((x @ w).reconstruct()),
        np.asarray(x.relu().reconstruct()),
        np.asarray(x.sigmoid().reconstruct()),
        float(np.asarray((big * 2).reconstruct()).sum()),
    )


def _check(testcase, out):
    xs = np.linspace(-3, 3, 64)
    mm, relu, ex, s = out
    np.testing.assert_allclose(mm, np.full(8, xs.mean()), atol=1e-2)
    np.testing.assert_allclose(relu, np.maximum(xs, 0), atol=1e-2)
    np.testing.assert_allclose(ex, 1 / (1 + np.exp(-xs)), atol=2e-2)
    testcase.assertAlmostEqual(s, 2 * 1500 * 1500, delta=1.0)


def _make_certs(tmp, n):
    """CA plus one certificate per process, via the openssl CLI."""
    def run(*args):
        subprocess.run(["openssl", *args], check=True, capture_output=True, cwd=tmp)

    run("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "ca.key", "-out", "ca.pem",
        "-days", "1", "-subj", "/CN=prix-test-ca",
        "-addext", "basicConstraints=critical,CA:TRUE",
        "-addext", "keyUsage=critical,keyCertSign,cRLSign",
        "-addext", "subjectKeyIdentifier=hash")
    # Python >= 3.13 verifies strictly (RFC 5280): leaf certs need these extensions.
    with open(os.path.join(tmp, "leaf.ext"), "w") as f:
        f.write("basicConstraints=critical,CA:FALSE\n"
                "keyUsage=critical,digitalSignature,keyEncipherment\n"
                "extendedKeyUsage=serverAuth,clientAuth\n"
                "subjectKeyIdentifier=hash\n"
                "authorityKeyIdentifier=keyid,issuer\n"
                "subjectAltName=IP:127.0.0.1,DNS:localhost\n")
    for i in range(n):
        run("req", "-newkey", "rsa:2048", "-nodes", "-keyout", f"p{i}.key", "-out", f"p{i}.csr",
            "-subj", f"/CN=prix-rank-{i}")
        run("x509", "-req", "-in", f"p{i}.csr", "-CA", "ca.pem", "-CAkey", "ca.key",
            "-CAcreateserial", "-out", f"p{i}.pem", "-days", "1", "-extfile", "leaf.ext")
    # All processes share one cert/key in this test; real deployments give each its own.
    return {"certfile": os.path.join(tmp, "p0.pem"), "keyfile": os.path.join(tmp, "p0.key"),
            "cafile": os.path.join(tmp, "ca.pem")}


class TestWire(unittest.TestCase):
    def roundtrip(self, obj):
        parts = wire.encode(obj)
        return wire.decode(bytearray(b"".join(bytes(p) for p in parts)))

    def test_roundtrip(self):
        arr = np.arange(12, dtype=np.int64).reshape(3, 4)
        msg = {"function": "f", "args": ((2, 3), None, True, -5, 2**70, 1.5, b"x"),
               "kwargs": {"device": "cpu"}, "a": arr, "s": np.int64(7), "l": [arr[0], np.float32(2)]}
        out = self.roundtrip(msg)
        np.testing.assert_array_equal(out["a"], arr)
        self.assertEqual(out["args"], ((2, 3), None, True, -5, 2**70, 1.5, b"x"))
        self.assertEqual(out["s"], 7)
        self.assertIsInstance(out["s"], np.int64)
        np.testing.assert_array_equal(out["l"][0], arr[0])

    def test_rejects_arbitrary_objects(self):
        with self.assertRaises(TypeError):
            wire.encode(object())
        with self.assertRaises(TypeError):
            wire.encode(np.array([object()]))

    def test_rejects_malformed_input(self):
        with self.assertRaises(ValueError):
            wire.decode(bytearray(b"Z"))
        good = bytearray(b"".join(bytes(p) for p in wire.encode(np.zeros(4))))
        with self.assertRaises(ValueError):
            wire.decode(good[:-1])


class TestTCPTransport(unittest.TestCase):
    def test_modes_over_tcp(self):
        for mode in ("crypten", "lut", "fission"):
            _check(self, prix.run(_workload, world_size=2, mode=mode, transport="tcp", timeout=300))

    def test_shared_memory_transport_still_works(self):
        _check(self, prix.run(_workload, world_size=2, mode="crypten", transport="shm", timeout=300))

    def test_ttp_dealer_over_tcp(self):
        prix.set_mode("crypten")
        with prix.cfg.temp_override({"mpc.provider": "TTP"}):
            _check(self, prix.run(_workload, world_size=2, transport="tcp", timeout=300))

    def test_three_parties_over_tcp(self):
        _check(self, prix.run(_workload, world_size=3, mode="crypten", transport="tcp", timeout=300))

    @unittest.skipIf(shutil.which("openssl") is None, "openssl CLI not available")
    def test_mutual_tls(self):
        with tempfile.TemporaryDirectory() as tmp:
            tls = _make_certs(tmp, 1)
            _check(self, prix.run(_workload, world_size=2, mode="crypten", transport="tcp",
                                  tls=tls, timeout=300))

    def test_manual_multi_host_api(self):
        """Two independent interpreters joined only by prix.init(peers=...), as on two hosts."""
        from prix.launcher import _free_ports

        peers = ",".join(f"127.0.0.1:{p}" for p in _free_ports(2))
        script = (
            "import sys, numpy as np, prix\n"
            "r = int(sys.argv[1])\n"
            f"prix.init(rank=r, world_size=2, mode='crypten', transport='tcp', peers='{peers}')\n"
            "x = prix.array(np.arange(6.0).reshape(2, 3) if r == 0 else None, src=0, broadcast_size=True)\n"
            "w = prix.array(np.ones((3, 2)) if r == 1 else None, src=1, broadcast_size=True)\n"
            "print('RESULT', np.asarray((x @ w).reconstruct()).round(2).tolist(), flush=True)\n"
            "prix.uninit()\n"
        )
        env = dict(os.environ, JAX_PLATFORMS="cpu")
        procs = [subprocess.Popen([sys.executable, "-c", script, str(r)], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, env=env) for r in range(2)]
        outs = [p.communicate(timeout=300) for p in procs]
        for p, (out, err) in zip(procs, outs):
            self.assertEqual(p.returncode, 0, err[-2000:])
            self.assertIn("RESULT [[3.0, 3.0], [12.0, 12.0]]", out)


if __name__ == "__main__":
    unittest.main()
