# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# Adapted for BLASt3R from COLMAP (https://github.com/colmap/colmap),
# scripts/python/read_write_model.py.
#
# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import struct
from pathlib import Path


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character='<'):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_images_binary(path):
    """Read a COLMAP `images.bin`, returning {image_name: (qvec, tvec)}."""
    images = {}
    with open(path, 'rb') as fid:
        num_reg_images = read_next_bytes(fid, 8, 'Q')[0]
        for _ in range(num_reg_images):
            properties = read_next_bytes(fid, num_bytes=64, format_char_sequence='idddddddi')
            qvec = properties[1:5]
            tvec = properties[5:8]

            name = b''
            char = read_next_bytes(fid, 1, 'c')[0]
            while char != b'\x00':  # look for the ASCII 0 entry
                name += char
                char = read_next_bytes(fid, 1, 'c')[0]

            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
            # the 2D observations are not needed, only the pose
            fid.seek(24 * num_points2D, 1)

            images[name.decode('utf-8')] = (qvec, tvec)

    return images


def read_images_text(path):
    """Read a COLMAP `images.txt`, returning {image_name: (qvec, tvec)}."""
    images = {}
    with open(path) as fid:
        lines = [line for line in fid if not line.startswith('#') and line.strip()]

    # every image occupies two lines: its pose, then its 2D observations
    for line in lines[::2]:
        fields = line.split()
        qvec = tuple(float(x) for x in fields[1:5])
        tvec = tuple(float(x) for x in fields[5:8])
        images[fields[-1]] = (qvec, tvec)

    return images


def read_images(model_dir):
    """Read the images of a COLMAP model, in whichever format it is stored.

    Returns {image_name: (qvec, tvec)}, where qvec is `(qw, qx, qy, qz)` and
    tvec the translation of the world-to-camera transform, as COLMAP stores
    them. Names are as recorded in the model, so they may include a directory.
    """
    model_dir = Path(model_dir)

    if (model_dir / 'images.bin').is_file():
        return read_images_binary(model_dir / 'images.bin')
    if (model_dir / 'images.txt').is_file():
        return read_images_text(model_dir / 'images.txt')

    raise FileNotFoundError(f'no images.bin or images.txt in {model_dir}')
