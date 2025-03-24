import gwyfile
import gsffile
import numpy as np
import pandas as pd
import os
import pathlib
import re


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

        if self.channelname == None:
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
            params[fieldname] = [float(ct[2]), float(ct[3])]

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
            channels = line.split("\t")
            channels = [channel.strip() for channel in channels[:-1]]

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

        if self._output == "dict":
            data = data.to_dict("list")
            for key in list(data.keys()):
                data[key] = np.asarray(data[key])

        return data, params


class ImageStackReader(Reader):
    """Reads a list of images from the subfolders of the specified folder by loading the files that contain the pattern string int the filename"""

    def __init__(self, folder=None):
        super().__init__(folder)
        self.folder = self.filename

    def read(self, pattern):
        imagestack = []
        wns = []
        filepaths = get_filenames(self.folder, pattern)

        for i, path in enumerate(filepaths):
            data_reader = GsfReader(path)
            imagestack.append(data_reader.read().data)

            try:
                txtpath = recreate_infofile_name_from_path(path)
                inforeader = NeaInfoReader(txtpath)
                infodict = inforeader.read()
                if infodict["TargetWavelength"] == "":
                    wn = infodict["InterferometerCenterDistance"][0]
                    wns.append(wn)
                else:
                    wn = infodict["TargetWavelength"]
                    if wn < 50.0:
                        wn = 10000 / wn
                    wns.append(wn)
            except:
                wns.append(i)
        
        idxs = np.argsort(np.asarray(wns))
        imagestack = [imagestack[i] for i in idxs]
        wns = [wns[i] for i in idxs]

        return imagestack, wns


def get_filenames(folder, pattern):
    """Returns the filepath of all files in the subfolders of the specified folder that contain pattern string in the filename"""

    filepaths = []

    for subfolder in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, subfolder)):
            for name in os.listdir(os.path.join(folder, subfolder)):
                if re.search(pattern, name):
                    subpath = os.path.join(subfolder, name)
                    filepaths.append(os.path.join(folder, subpath))

    return filepaths


def recreate_infofile_name_from_path(filepath):
    """Recreates the name of the info file from the path of the data file"""

    pathparts = list(pathlib.PurePath(filepath).parts)
    newparts = pathparts[:-1]
    newparts.append(pathparts[-2] + ".txt")

    return str(pathlib.PurePath(*newparts))


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
                runs = [1]

            indexes = np.unique(meta["Channel"], return_index=True)[1]
            channels = [meta["Channel"][index] for index in sorted(indexes)]
            channels = [channel.decode("utf-8") for channel in channels]

            for name in h:
                if name != "Channel":
                    data[name] = np.array(meta[name])

            for i in range(len(channels)):
                data[channels[i]] = np.ravel(
                    C_data[i * len(runs) : (i + 1) * len(runs), :]
                )

            params["PixelArea"] = [int(data["Row"].max() + 1), int(data["Column"].max() + 1), np.shape(C_data)[1]]
            params["Scan"] = "Fourier Scan"

        return data, params
