"""
Which files the HTTP interface will act on.

Every path the interface accepts comes from an upload it answered
earlier, so every path it accepts must be inside the workspace it wrote.
The two-step flow checked that; the older shape -- a `vocal_path` plus
catalog beat ids -- took any path that existed, which let a caller name a
file anywhere on the machine, have it analysed, and have its audio
rendered into a file the server then served back. These tests pin the
check on all three shapes, and the 422 on an answer the engine does not
recognise.

The server is exercised in-process; no job is allowed to reach a render.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import soundfile as sf
    from fastapi.testclient import TestClient
    from mixengine.api.app import create_app
    HAVE_API = True
except Exception:                                        # pragma: no cover
    HAVE_API = False


@unittest.skipUnless(HAVE_API, "needs fastapi, httpx and soundfile")
class TestPathsMustBeUploads(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        cls.client = TestClient(create_app(os.path.join(cls.root, "data")))
        # A file the server never wrote, outside the workspace.
        cls.outside = os.path.join(cls.root, "elsewhere.wav")
        sf.write(cls.outside, np.zeros(800, dtype="float32"), 8000)
        # A file inside the workspace, as an upload would have left it.
        inside_dir = os.path.join(cls.root, "data", "vocals", "abc123")
        os.makedirs(inside_dir, exist_ok=True)
        cls.inside = os.path.join(inside_dir, "take.wav")
        sf.write(cls.inside, np.zeros(800, dtype="float32"), 8000)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def post(self, url, **data):
        return self.client.post(url, data=data)

    def test_the_older_shape_refuses_a_path_it_did_not_write(self):
        r = self.post("/api/render", vocal_path=self.outside, beat_ids="b1")
        self.assertEqual(r.status_code, 400)
        self.assertIn("not an uploaded file", r.json()["detail"])

    def test_the_older_shape_refuses_a_traversal_out_of_the_workspace(self):
        sneaky = os.path.join(self.root, "data", "vocals", "..", "..",
                              "elsewhere.wav")
        r = self.post("/api/render", vocal_path=sneaky, beat_ids="b1")
        self.assertEqual(r.status_code, 400)

    def test_the_older_shape_accepts_a_path_inside_the_workspace(self):
        """It gets as far as a job. The job then fails on a silent file,
        which is the analysis talking, not the path check."""
        r = self.post("/api/render", vocal_path=self.inside, beat_ids="")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("id", r.json())

    def test_the_two_step_shape_refuses_a_path_it_did_not_write(self):
        r = self.post("/api/render", vocal_path=self.outside,
                      beat_path=self.outside)
        self.assertEqual(r.status_code, 400)

    def test_preparing_refuses_a_path_it_did_not_write(self):
        r = self.post("/api/prepare", vocal_path=self.inside,
                      beat_path=self.outside)
        self.assertEqual(r.status_code, 400)

    def test_preparing_needs_both_files(self):
        self.assertEqual(self.post("/api/prepare").status_code, 400)

    def test_an_answer_the_engine_does_not_know_is_rejected(self):
        r = self.post("/api/render", vocal_path=self.inside, beat_ids="b1",
                      performance="opera")
        self.assertEqual(r.status_code, 422)
        self.assertIn("performance", r.json()["detail"])

    def test_the_refusal_answer_is_accepted_as_a_value(self):
        """"rerecord" is what a blocking question defaults to; posting it
        must not look like a bad request."""
        r = self.post("/api/render", vocal_path=self.inside, beat_ids="",
                      noise="rerecord", length="rerecord")
        self.assertEqual(r.status_code, 200, r.text)


if __name__ == "__main__":
    unittest.main()
