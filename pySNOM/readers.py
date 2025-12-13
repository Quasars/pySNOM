import gwyfile
import gsffile
import numpy as np
import pandas as pd
import os
from pathlib import PurePath
import re
from pySNOM.images import Transformation


class Reader:
    def __init__(self, fullfilepath=None):
        self.filename = fullfilepath


class GwyReader(Reader):
    def __init__(self, fullfilepath=None, channelname=None):
        super().__init__(fullfilepath)
        self.channelname = channelname

    def read(self):
        # Returns a dictionary of all the channels
        gwyobj = gwyfile.load(self.filename)
        allchannels = gwyfile.util.get_datafields(gwyobj)

        if self.channelname is None:
            return allchannels
        else:
            # Read channels from gwyfile and return only a specific one
            channel = allchannels[self.channelname]
            return channel


class GsfReader(Reader):
    def __init__(self, fullfilepath=None):
        super().__init__(fullfilepath)

    def read(self):
        data, metadata = gsffile.read_gsf(self.filename)
        channel = gwyfile.objects.GwyDataField(
            data,
            xreal=metadata["XReal"],
            yreal=metadata["YReal"],
            xoff=metadata["XOffset"],
            yoff=metadata["YOffset"],
            si_unit_xy=None,
            si_unit_z=None,
            typecodes=None,
        )
        return channel


class NeaHeaderReader(Reader):
    def __init__(self, fullfilepath=None):
        super().__init__(fullfilepath)

    @staticmethod
    def parseline(linestring, params={}):
        ct = linestring.split("\t")
        fieldname = ct[0][2:-1]
        fieldname = fieldname.replace(" ", "")

        if "Scanner Center Position" in linestring:
            fieldname = fieldname[:-5]
            params[fieldname] = [float(ct[2]), float(ct[3])]

        elif "Scan Area" in linestring:
            fieldname = fieldname[:-7]
            params[fieldname] = [float(ct[2]), float(ct[3]), float(ct[4])]

        elif "Pixel Area" in linestring:
            fieldname = fieldname[:-7]
            params[fieldname] = [int(ct[2]), int(ct[3]), int(ct[4])]

        elif "Averaging" in linestring:
            params[fieldname] = int(ct[2])

        elif "Interferometer Center/Distance" in linestring:
            fieldname = fieldname.replace("/", "")
            params[fieldname] = [
                float(ct[2].replace(",", "")),
                float(ct[3].replace(",", "")),
            ]

        elif "Regulator" in linestring:
            fieldname = fieldname[:-7]
            params[fieldname] = [float(ct[2]), float(ct[3]), float(ct[4])]

        elif "Q-Factor" in linestring:
            fieldname = fieldname.replace("-", "")
            params[fieldname] = float(ct[2])

        else:
            fieldname = ct[0][2:-1]
            fieldname = fieldname.replace(" ", "")
            val = ct[2]
            val = val.replace(",", "")
            try:
                params[fieldname] = float(val)
            except:
                params[fieldname] = val.strip()

        return params

    def read(self):
        params = {}
        with open(self.filename, encoding="utf8") as f:
            # Read www.neaspec.com
            line = f.readline()
            count = 1
            while f:
                line = f.readline()
                count = count + 1
                try:
                    if line[0] not in ("#", "\n"):
                        break
                    if line[0] == "#":
                        params = NeaHeaderReader.parseline(line, params)
                except IndexError:
                    break

            channels = line.strip().split("\t")
            channels = [channel.strip() for channel in channels]

        return channels, params


class NeaInfoReader(NeaHeaderReader):
    def __init__(self, fullfilepath=None):
        super().__init__(fullfilepath)

    def read(self):
        _, infodict = super().read()
        return infodict


class NeaSpectralReader(Reader):
    def __init__(self, fullfilepath=None, output="dict"):
        super().__init__(fullfilepath)
        self._output = output

    def read(self):
        data = {}

        channels, params = NeaHeaderReader(self.filename).read()
        channels.append("")

        count = len(list(params.keys())) + 2

        data = pd.read_csv(
            self.filename,
            sep="\t",
            skiprows=count,
            encoding="utf-8",
            names=channels,
            lineterminator="\n",
        ).dropna(axis=1, how="all")

        cols_to_keep = [c for c in data.columns if c != ""]
        data = data[cols_to_keep]

        xpos, ypos = PositionTransformer().transform(data, params)

        data.insert(2, "X", xpos)
        data.insert(3, "Y", ypos)

        if self._output == "dict":
            data = data.to_dict("list")
            for key in list(data.keys()):
                data[key] = np.asarray(data[key])

        return data, params


class ImageStackReader(Reader):
    """Reads a list of images from the subfolders of the specified folder by loading the files that contain the pattern string in the filename"""

    def __init__(self, folder=None, folder_pattern=""):
        super().__init__(folder)
        self.folder = self.filename
        self.folder_pattern = folder_pattern

    def read(self, pattern):
        imagestack = []
        wns = []
        filepaths = get_filenames(
            self.folder, pattern, folderpattern=self.folder_pattern
        )

        for i, path in enumerate(filepaths):
            data_reader = GsfReader(path)
            imagestack.append(data_reader.read().data)

            try:
                txtpath = recreate_infofile_name_from_path(path)
                inforeader = NeaInfoReader(txtpath)
                infodict = inforeader.read()
                wn = get_wl_from_infofile(infodict)
                wns.append(wn)
            except:
                wns.append(i)

        idxs = np.argsort(np.asarray(wns))
        imagestack = [imagestack[i] for i in idxs]
        wns = [wns[i] for i in idxs]

        return imagestack, wns


def get_wl_from_infofile(infodict: dict):
    """Returns the wavelength from the info file. If not found, returns 0.0"""
    try:
        if infodict["TargetWavelength"] == "":
            wn = infodict["InterferometerCenterDistance"][0]
        else:
            wn = infodict["TargetWavelength"]
            # NeaSpec is not consistent in the units of the wavelength.
            # Sometimes it is in cm-1, sometimes in um.
            if wn < 50.0:
                wn = 10000 / wn
    except:
        wn = None
    return wn


def get_wl_from_filename(filename):
    """Returns the wavelength from the filename. If not found, returns None"""

    wn = re.findall(
        r"(?<=[_-])(\d+(?:\.\d+)?)(?=(?:_?cm[-_]1|-?cm[-_]1))", PurePath(filename).name
    )

    if wn:
        wn = float(wn[0])
    else:
        wn = None

    return wn


def get_filenames(folder: str, pattern: str, folderpattern=""):
    """Returns the filepath of all files in the subfolders of the specified folder that contain pattern string in the filename"""

    filepaths = []

    for subfolder in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, subfolder)) and re.search(
            folderpattern, subfolder
        ):
            for name in os.listdir(os.path.join(folder, subfolder)):
                if re.search(pattern, name):
                    subpath = os.path.join(subfolder, name)
                    filepaths.append(os.path.join(folder, subpath))

    return filepaths


def recreate_infofile_name_from_path(filepath: str):
    """Recreates the name of the info file from the path of the data file"""

    pathparts = list(PurePath(filepath).parts)
    newparts = pathparts[:-1]
    newparts.append(pathparts[-2] + ".txt")

    return str(PurePath(*newparts))


class NeaFileLegacyReader(Reader):
    """Reader for .nea files from older neasnom microscopes"""

    def __init__(self, fullfilepath=None):
        super().__init__(fullfilepath)

    def read(self):
        data = {}
        params = {}

        with open(self.filename, encoding="utf8") as f:
            h = next(f)  # header
            h = h.strip()
            h = h.split("\t")

            l = next(f)
            l = l.strip()
            l = l.split("\t")

            f.seek(0)
            next(f)
            datacols = np.arange(len(h), len(l))
            C_data = np.loadtxt(f, dtype="float", usecols=datacols)

            f.seek(0)
            next(f)

            metacols = np.arange(0, 4)
            meta = np.loadtxt(
                f,
                dtype={"names": tuple(h), "formats": (int, int, int, "S10")},
                usecols=metacols,
            )
            if "Run" in h:
                runs = np.unique(meta["Run"])
            else:
                runs = [0]

            Max_row = len(np.unique(meta["Row"]))
            Max_col = len(np.unique(meta["Column"]))
            Max_run = len(runs)
            Max_omega = np.shape(C_data)[1]

            N_rows = Max_row * Max_col * Max_run * Max_omega

            indexes = np.unique(meta["Channel"], return_index=True)[1]
            channels = [meta["Channel"][index] for index in sorted(indexes)]
            channels = [channel.decode("utf-8") for channel in channels]

            for name in h:
                if name != "Channel":
                    data[name] = np.array(meta[name])

            for i in range(len(channels)):
                data[channels[i]] = np.ravel(C_data[i * Max_run : (i + 1) * Max_run, :])

        alpha = 0
        beta = 0
        data["Run"] = np.zeros(N_rows)
        data["Column"] = np.zeros(N_rows)
        data["Row"] = np.zeros(N_rows)

        for i in range(0, N_rows, Max_omega * Max_run):
            if beta == Max_row:
                beta = 0
                alpha = alpha + 1
            data["Run"][i : i + Max_omega * Max_run] = np.repeat(
                np.arange(Max_run), Max_omega
            )
            data["Column"][i : i + Max_omega * Max_run] = alpha
            data["Row"][i : i + Max_omega * Max_run] = beta
            beta = beta + 1

            params["PixelArea"] = [
                Max_row,
                Max_col,
                Max_omega,
            ]
            params["Scan"] = "Fourier Scan"

        return data, params


class PositionTransformer(Transformation):
    def transform(self, data, measparams):
        Max_row = len(np.unique(data["Row"]))
        Max_col = len(np.unique(data["Column"]))

        if "Depth" in data:
            Max_omega = len(np.unique(data["Depth"]))
        elif "Index" in data:
            Max_omega = len(np.unique(data["Index"]))
        elif "Omega" in data:
            Max_omega = len(np.unique(data["Omega"]))
        else:
            raise ValueError("Variable index not found")

        if "Run" in data:
            Max_run = len(np.unique(data["Run"]))
        else:
            Max_run = 1

        # Calculate coordinates for each point if parameters are given
        if "Rotation" in measparams:
            angle = np.radians(measparams["Rotation"])
        else:
            angle = 0
        if "ScanArea" in measparams:
            width = measparams["ScanArea"][0]
            height = measparams["ScanArea"][1]
        else:
            width = Max_col if Max_col > 1 else 0
            height = Max_row if Max_row > 1 else 0
        if "ScannerCenterPosition" in measparams:
            xoff = measparams["ScannerCenterPosition"][0]
            yoff = measparams["ScannerCenterPosition"][1]
        else:
            xoff = 0.0
            yoff = 0.0

        # Create the list of points centered to the origo
        X, Y = np.meshgrid(
            np.linspace(-width / 2, width / 2, Max_col),
            np.linspace(-height / 2, height / 2, Max_row),
        )
        xvec = X.ravel()
        yvec = Y.ravel()
        xpos = []
        ypos = []
        # Rotated the coordinate system to match orientation
        c, s = np.cos(angle), np.sin(angle)
        R = np.array(((c, -s), (s, c)))
        for i in range(len(xvec)):
            vec = np.array([xvec[i], yvec[i]])
            vec = np.matmul(R, vec)
            vec[0] += xoff
            vec[1] += yoff
            xpos.append(vec[0])
            ypos.append(vec[1])
        # Reshape
        xpos = np.reshape(np.array(xpos), (Max_col, Max_row))
        ypos = np.reshape(np.array(ypos), (Max_col, Max_row))

        alpha = 0
        beta = 0

        x = np.zeros((Max_row * Max_col * Max_omega * Max_run, 1))
        y = np.zeros((Max_row * Max_col * Max_omega * Max_run, 1))

        for i in range(0, Max_row * Max_col * Max_omega * Max_run, Max_omega * Max_run):
            if beta == Max_row:
                beta = 0
                alpha = alpha + 1

            for jrun in range(Max_run):
                idx = slice(i + jrun * Max_omega, i + (jrun + 1) * Max_omega)
                x[idx] = xpos[alpha, beta]
                y[idx] = ypos[alpha, beta]

            beta = beta + 1

        return x, y
