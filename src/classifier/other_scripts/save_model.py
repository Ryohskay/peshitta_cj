# BSD 2-Clause License
#
# Copyright (c) 2025, Ryosuke Nagata
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""A script to bridge between model implementations and ONNX platform."""

from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int32TensorType
from sklearn.base import BaseEstimator


def save_to_onnx(
    clf: BaseEstimator,
    save_file_name: Path | str = "./out/classifier_saved.onnx",
) -> None:
    """Convert the given classifier into ONNX format and save into a file.

    Code in this function has been adapted from: `ONNX tutorial <https://onnx.ai/sklearn-onnx/introduction.html>`
    (Accessed: 8 April 2025).

    Args:
        clf: any scikit-learn estimator object supported by sklearn-ONNX.
            See `sklearn-ONNX library's documentation <https://onnx.ai/sklearn-onnx/supported.html>`
            for the supported kinds of sklearn estimators.
        save_file_name: a :class:`python:pathlib.Path` object or string
            indicating path to save the model in the ONNX-format.

    Raises:
        ValueError: if the ``save_file_name`` path does not exist.
    """
    initial_type = [("Int32 inputs", Int32TensorType([None, 4]))]
    onx = convert_sklearn(clf, initial_types=initial_type)
    if not Path(save_file_name).exists():
        msg = (
            f"The provided file path {save_file_name} for save file "
            + "could not be reached."
        )
        raise ValueError(msg)
    Path(save_file_name).write_bytes(onx.SerializeToString())
