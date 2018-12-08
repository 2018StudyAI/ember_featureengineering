#!/usr/bin/python
''' Extracts some basic features from PE files. Many of the features
implemented have been used in previously published works. For more information,
check out the following resources:
* Schultz, et al., 2001: http://128.59.14.66/sites/default/files/binaryeval-ieeesp01.pdf
* Kolter and Maloof, 2006: http://www.jmlr.org/papers/volume7/kolter06a/kolter06a.pdf
* Shafiq et al., 2009: https://www.researchgate.net/profile/Fauzan_Mirza/publication/242084613_A_Framework_for_Efficient_Mining_of_Structural_Information_to_Detect_Zero-Day_Malicious_Portable_Executables/links/0c96052e191668c3d5000000.pdf
* Raman, 2012: http://2012.infosecsouthwest.com/files/speaker_materials/ISSW2012_Selecting_Features_to_Classify_Malware.pdf
* Saxe and Berlin, 2015: https://arxiv.org/pdf/1508.03096.pdf

It may be useful to do feature selection to reduce this set of features to a meaningful set
for your modeling problem.
'''

import re
import lief
import hashlib
import numpy as np
from sklearn.feature_extraction import FeatureHasher
import time
import pandas as pd
import pefile


class FeatureType(object):
    ''' Base class from which each feature type may inherit '''

    name = ''
    dim = 0

    def __repr__(self):
        return '{}({})'.format(self.name, self.dim)

    def raw_features(self, bytez, lief_binary, pefile_binary):
        ''' Generate a JSON-able representation of the file '''
        raise (NotImplemented)

    def process_raw_features(self, raw_obj):
        ''' Generate a feature vector from the raw features '''
        raise (NotImplemented)

    # def feature_vector(self, bytez, lief_binary, pefile_binary):
    #     ''' Directly calculate the feature vector from the sample itself. This should only be implemented differently
    #     if there are significant speedups to be gained from combining the two functions. '''
    #     return self.process_raw_features(self.raw_features(bytez, lief_binary, pefile_binary))


class ByteHistogram(FeatureType):
    ''' Byte histogram (count + non-normalized) over the entire binary file '''

    name = 'histogram'
    dim = 256

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary, pefile_binary):
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        return counts.tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized


class ByteEntropyHistogram(FeatureType):
    ''' 2d byte/entropy histogram based loosely on (Saxe and Berlin, 2015).
    This roughly approximates the joint probability of byte value and local entropy.
    See Section 2.1.1 in https://arxiv.org/pdf/1508.03096.pdf for more info.
    '''

    name = 'byteentropy'
    dim = 256

    def __init__(self, step=1024, window=2048):
        super(FeatureType, self).__init__()
        self.window = window
        self.step = step

    def _entropy_bin_counts(self, block):
        # coarse histogram, 16 bytes per bin
        c = np.bincount(block >> 4, minlength=16)  # 16-bin histogram
        p = c.astype(np.float32) / self.window
        wh = np.where(c)[0]
        H = np.sum(-p[wh] * np.log2(
            p[wh])) * 2  # * x2 b.c. we reduced information by half: 256 bins (8 bits) to 16 bins (4 bits)

        Hbin = int(H * 2)  # up to 16 bins (max entropy is 8 bits)
        if Hbin == 16:  # handle entropy = 8.0 bits
            Hbin = 15

        return Hbin, c

    def raw_features(self, bytez, lief_binary, pefile_binary):
        output = np.zeros((16, 16), dtype=np.int)
        a = np.frombuffer(bytez, dtype=np.uint8)
        if a.shape[0] < self.window:
            Hbin, c = self._entropy_bin_counts(a)
            output[Hbin, :] += c
        else:
            # strided trick from here: http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
            shape = a.shape[:-1] + (a.shape[-1] - self.window + 1, self.window)
            strides = a.strides + (a.strides[-1],)
            blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::self.step, :]

            # from the blocks, compute histogram
            for block in blocks:
                Hbin, c = self._entropy_bin_counts(block)
                output[Hbin, :] += c

        return output.flatten().tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized


class SectionInfo(FeatureType):
    ''' Information about section names, sizes and entropy.  Uses hashing trick
    to summarize all this section info into a feature vector.
    '''

    name = 'section'
    dim = 5 + 50 + 50 + 50 + 50 + 50

    def __init__(self):
        super(FeatureType, self).__init__()

    @staticmethod
    def _properties(s):
        return [str(c).split('.')[-1] for c in s.characteristics_lists]

    def raw_features(self, bytez, lief_binary, pefile_binary):
        if lief_binary is None:
            return {"entry": "", "sections": []}

        # properties of entry point, or if invalid, the first executable section
        try:
            entry_section = lief_binary.section_from_offset(lief_binary.entrypoint).name
        except lief.not_found:
            # bad entry point, let's find the first executable section
            entry_section = ""
            for s in lief_binary.sections:
                if lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE in s.characteristics_lists:
                    entry_section = s.name
                    break

        raw_obj = {"entry": entry_section}
        raw_obj["sections"] = [{
            'name': s.name,
            'size': s.size,
            'entropy': s.entropy,
            'vsize': s.virtual_size,
            'props': self._properties(s)
        } for s in lief_binary.sections]
        return raw_obj

    def process_raw_features(self, raw_obj):
        sections = raw_obj['sections']
        general = [
            len(sections),  # total number of sections
            # number of sections with nonzero size
            sum(1 for s in sections if s['size'] == 0),
            # number of sections with an empty name
            sum(1 for s in sections if s['name'] == ""),
            # number of RX
            sum(1 for s in sections if 'MEM_READ' in s['props'] and 'MEM_EXECUTE' in s['props']),
            # number of W
            sum(1 for s in sections if 'MEM_WRITE' in s['props'])
        ]
        # gross characteristics of each section
        section_sizes = [(s['name'], s['size']) for s in sections]
        section_sizes_hashed = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
        section_entropy = [(s['name'], s['entropy']) for s in sections]
        section_entropy_hashed = FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0]
        section_vsize = [(s['name'], s['vsize']) for s in sections]
        section_vsize_hashed = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
        entry_name_hashed = FeatureHasher(50, input_type="string").transform([raw_obj['entry']]).toarray()[0]
        characteristics = [p for s in sections for p in s['props'] if s['name'] == raw_obj['entry']]
        characteristics_hashed = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]

        return np.hstack([
            general, section_sizes_hashed, section_entropy_hashed, section_vsize_hashed, entry_name_hashed,
            characteristics_hashed
        ]).astype(np.float32)


class ImportsInfo(FeatureType):
    ''' Information about imported libraries and functions from the
    import address table.  Note that the total number of imported
    functions is contained in GeneralFileInfo.
    '''

    name = 'imports'
    dim = 1280

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary, pefile_binary):
        imports = {}
        if lief_binary is None:
            return imports

        for lib in lief_binary.imports:
            if lib.name not in imports:
                imports[lib.name] = []  # libraries can be duplicated in listing, extend instead of overwrite

            # Clipping assumes there are diminishing returns on the discriminatory power of imported functions
            #  beyond the first 10000 characters, and this will help limit the dataset size
            imports[lib.name].extend([entry.name[:10000] for entry in lib.entries])

        return imports

    def process_raw_features(self, raw_obj):
        # unique libraries
        libraries = list(set([l.lower() for l in raw_obj.keys()]))
        libraries_hashed = FeatureHasher(256, input_type="string").transform([libraries]).toarray()[0]

        # A string like "kernel32.dll:CreateFileMappingA" for each imported function
        imports = [lib.lower() + ':' + e for lib, elist in raw_obj.items() for e in elist]
        imports_hashed = FeatureHasher(1024, input_type="string").transform([imports]).toarray()[0]

        # Two separate elements: libraries (alone) and fully-qualified names of imported functions
        return np.hstack([libraries_hashed, imports_hashed]).astype(np.float32)


class ExportsInfo(FeatureType):
    ''' Information about exported functions. Note that the total number of exported
    functions is contained in GeneralFileInfo.
    '''

    name = 'exports'
    dim = 128

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary, pefile_binary):
        if lief_binary is None:
            return []

        # Clipping assumes there are diminishing returns on the discriminatory power of exports beyond
        #  the first 10000 characters, and this will help limit the dataset size
        clipped_exports = [export[:10000] for export in lief_binary.exported_functions]

        return clipped_exports

    def process_raw_features(self, raw_obj):
        exports_hashed = FeatureHasher(128, input_type="string").transform([raw_obj]).toarray()[0]
        return exports_hashed.astype(np.float32)


class GeneralFileInfo(FeatureType):
    ''' General information about the file '''

    name = 'general'
    dim = 10

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary, pefile_binary):
        if lief_binary is None:
            return {
                'size': len(bytez),
                'vsize': 0,
                'has_debug': 0,
                'exports': 0,
                'imports': 0,
                'has_relocations': 0,
                'has_resources': 0,
                'has_signature': 0,
                'has_tls': 0,
                'symbols': 0
            }

        return {
            'size': len(bytez),
            'vsize': lief_binary.virtual_size,
            'has_debug': int(lief_binary.has_debug),
            'exports': len(lief_binary.exported_functions),
            'imports': len(lief_binary.imported_functions),
            'has_relocations': int(lief_binary.has_relocations),
            'has_resources': int(lief_binary.has_resources),
            'has_signature': int(lief_binary.has_signature),
            'has_tls': int(lief_binary.has_tls),
            'symbols': len(lief_binary.symbols),
        }

    def process_raw_features(self, raw_obj):
        return np.asarray(
            [
                raw_obj['size'], raw_obj['vsize'], raw_obj['has_debug'], raw_obj['exports'], raw_obj['imports'],
                raw_obj['has_relocations'], raw_obj['has_resources'], raw_obj['has_signature'], raw_obj['has_tls'],
                raw_obj['symbols']
            ],
            dtype=np.float32)


class HeaderFileInfo(FeatureType):
    ''' Machine, architecure, OS, linker and other information extracted from header '''

    name = 'header'
    dim = 37

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary, pefile_binary):
        raw_obj = {}
        dos_header = pefile_binary.DOS_HEADER
        file_header = pefile_binary.FILE_HEADER
        optional_header = pefile_binary.OPTIONAL_HEADER
        
        raw_obj['dos'] = {
            'e_magic': 0,
            'e_lfanew': 0,
        }
        raw_obj['coff'] = {
            'numberofsections': 0,
            'timedatestamp': 0,
            'pointertosymboltable': 0,
            'numberofsymbols': 0,
            'sizeofoptionalhedaer': 0,
            'characteristics': 0
        }        
        raw_obj['optional'] = {
            'magic': 0,
            'majorlinkerversion': 0,
            'minorlinkerversion': 0,
            'sizeofcode': 0,
            'sizeofinitalizedata': 0,
            'sizeofuninitalizedata': 0,
            'addressofentrypoint': 0,
            'baseofcode': 0,
            'baseofdata': 0,
            'imagebase': 0,
            'sectionalignment': 0,
            'filealignment': 0,
            'majoroperatingsystemversion': 0,
            'minoroperatingsystemversion': 0,
            'majorimageversion': 0,
            'minorimageversion': 0,
            'majorsubystemversion': 0,
            'minorsubsystemversion': 0,
            'reversed1': 0,
            'sizeofimage': 0,
            'sizeofheader': 0,
            'checksum': 0,
            'dllcharacteristics': 0,
            'sizeofstackreserve': 0,
            'sizeofstackcommit': 0,
            'sizeofheapreserve': 0,
            'sizeofheapcommit': 0,
            'loadflags': 0,
            'numberofrvaandsizes': 0
        }

        if lief_binary is None:
            return raw_obj
        
        raw_obj['dos']['e_magic'] = dos_header.e_magic
        raw_obj['dos']['e_lfanew'] = dos_header.e_lfanew
        raw_obj['coff']['numberofsections'] = file_header.NumberOfSections
        raw_obj['coff']['timedatestamp'] = file_header.TimeDateStamp
        raw_obj['coff']['pointertosymboltable'] = file_header.PointerToSymbolTable
        raw_obj['coff']['numberofsymbols'] = file_header.NumberOfSymbols
        raw_obj['coff']['sizeofoptionalhedaer'] = file_header.SizeOfOptionalHeader
        raw_obj['coff']['characteristics'] = file_header.Characteristics
        raw_obj['optional']['magic'] = optional_header.Magic
        raw_obj['optional']['majorlinkerversion'] = optional_header.MajorLinkerVersion
        raw_obj['optional']['minorlinkerversion'] = optional_header.MinorLinkerVersion
        raw_obj['optional']['sizeofcode'] = optional_header.SizeOfCode
        raw_obj['optional']['sizeofinitalizedata'] = optional_header.SizeOfInitializedData
        raw_obj['optional']['sizeofuninitalizedata'] = optional_header.SizeOfUninitializedData
        raw_obj['optional']['addressofentrypoint'] = optional_header.AddressOfEntryPoint
        raw_obj['optional']['baseofcode'] = optional_header.BaseOfCode
        raw_obj['optional']['baseofdata'] = optional_header.BaseOfData
        raw_obj['optional']['imagebase'] = optional_header.ImageBase
        raw_obj['optional']['sectionalignment'] = optional_header.SectionAlignment
        raw_obj['optional']['filealignment'] = optional_header.FileAlignment
        raw_obj['optional']['majoroperatingsystemversion'] = optional_header.MajorOperatingSystemVersion
        raw_obj['optional']['minoroperatingsystemversion'] = optional_header.MinorOperatingSystemVersion
        raw_obj['optional']['majorimageversion'] = optional_header.MajorImageVersion
        raw_obj['optional']['minorimageversion'] = optional_header.MinorImageVersion
        raw_obj['optional']['majorsubystemversion'] = optional_header.MajorSubsystemVersion
        raw_obj['optional']['minorsubsystemversion'] = optional_header.MinorSubsystemVersion
        raw_obj['optional']['reversed1'] = optional_header.Reserved1
        raw_obj['optional']['sizeofimage'] = optional_header.SizeOfImage
        raw_obj['optional']['sizeofheader'] = optional_header.SizeOfHeaders
        raw_obj['optional']['checksum'] = optional_header.CheckSum
        raw_obj['optional']['dllcharacteristics'] = optional_header.DllCharacteristics
        raw_obj['optional']['sizeofstackreserve'] = optional_header.SizeOfStackReserve
        raw_obj['optional']['sizeofstackcommit'] = optional_header.SizeOfStackCommit
        raw_obj['optional']['sizeofheapreserve'] = optional_header.SizeOfHeapReserve
        raw_obj['optional']['sizeofheapcommit'] = optional_header.SizeOfHeapCommit
        raw_obj['optional']['loadflags'] = optional_header.LoaderFlags
        raw_obj['optional']['numberofrvaandsizes'] = optional_header.NumberOfRvaAndSizes

        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            raw_obj['dos']['e_magic'],
            raw_obj['dos']['e_lfanew'],
            raw_obj['coff']['numberofsections'],
            raw_obj['coff']['timedatestamp'],
            raw_obj['coff']['pointertosymboltable'],
            raw_obj['coff']['numberofsymbols'],
            raw_obj['coff']['sizeofoptionalhedaer'],
            raw_obj['coff']['characteristics'],
            raw_obj['optional']['magic'],
            raw_obj['optional']['majorlinkerversion'],
            raw_obj['optional']['minorlinkerversion'],
            raw_obj['optional']['sizeofcode'],
            raw_obj['optional']['sizeofinitalizedata'],
            raw_obj['optional']['sizeofuninitalizedata'],
            raw_obj['optional']['addressofentrypoint'],
            raw_obj['optional']['baseofcode'],
            raw_obj['optional']['baseofdata'],
            raw_obj['optional']['imagebase'],
            raw_obj['optional']['sectionalignment'],
            raw_obj['optional']['filealignment'],
            raw_obj['optional']['majoroperatingsystemversion'],
            raw_obj['optional']['minoroperatingsystemversion'],
            raw_obj['optional']['majorimageversion'],
            raw_obj['optional']['minorimageversion'],
            raw_obj['optional']['majorsubystemversion'],
            raw_obj['optional']['minorsubsystemversion'],
            raw_obj['optional']['reversed1'],
            raw_obj['optional']['sizeofimage'],
            raw_obj['optional']['sizeofheader'],
            raw_obj['optional']['checksum'],
            raw_obj['optional']['dllcharacteristics'],
            raw_obj['optional']['sizeofstackreserve'],
            raw_obj['optional']['sizeofstackcommit'],
            raw_obj['optional']['sizeofheapreserve'],
            raw_obj['optional']['sizeofheapcommit'],
            raw_obj['optional']['loadflags'],
            raw_obj['optional']['numberofrvaandsizes']
        ]).astype(np.float32)


class StringExtractor(FeatureType):
    ''' Extracts strings from raw byte stream '''

    name = 'strings'
    dim = 1 + 1 + 1 + 96 + 1 + 1 + 1 + 1 + 1

    def __init__(self):
        super(FeatureType, self).__init__()
        # all consecutive runs of 0x20 - 0x7f that are 5+ characters
        self._allstrings = re.compile(b'[\x20-\x7f]{5,}')
        # occurances of the string 'C:\'.  Not actually extracting the path
        self._paths = re.compile(b'c:\\\\', re.IGNORECASE)
        # occurances of http:// or https://.  Not actually extracting the URLs
        self._urls = re.compile(b'https?://', re.IGNORECASE)
        # occurances of the string prefix HKEY_.  No actually extracting registry names
        self._registry = re.compile(b'HKEY_')
        # crude evidence of an MZ header (dropper?) somewhere in the byte stream
        self._mz = re.compile(b'MZ')

    def raw_features(self, bytez, lief_binary, pefile_binary):
        allstrings = self._allstrings.findall(bytez)
        if allstrings:
            # statistics about strings:
            string_lengths = [len(s) for s in allstrings]
            avlength = sum(string_lengths) / len(string_lengths)
            # map printable characters 0x20 - 0x7f to an int array consisting of 0-95, inclusive
            as_shifted_string = [b - ord(b'\x20') for b in b''.join(allstrings)]
            c = np.bincount(as_shifted_string, minlength=96)  # histogram count
            # distribution of characters in printable strings
            csum = c.sum()
            p = c.astype(np.float32) / csum
            wh = np.where(c)[0]
            H = np.sum(-p[wh] * np.log2(p[wh]))  # entropy
        else:
            avlength = 0
            c = np.zeros((96,), dtype=np.float32)
            H = 0
            csum = 0

        return {
            'numstrings': len(allstrings),
            'avlength': avlength,
            'printabledist': c.tolist(),  # store non-normalized histogram
            'printables': int(csum),
            'entropy': float(H),
            'paths': len(self._paths.findall(bytez)),
            'urls': len(self._urls.findall(bytez)),
            'registry': len(self._registry.findall(bytez)),
            'MZ': len(self._mz.findall(bytez))
        }

    def process_raw_features(self, raw_obj):
        hist_divisor = float(raw_obj['printables']) if raw_obj['printables'] > 0 else 1.0
        return np.hstack([
            raw_obj['numstrings'], raw_obj['avlength'], raw_obj['printables'],
            np.asarray(raw_obj['printabledist']) / hist_divisor, raw_obj['entropy'], raw_obj['paths'], raw_obj['urls'],
            raw_obj['registry'], raw_obj['MZ']
        ]).astype(np.float32)


def GenerateTime(lief_binary):
    fileheader = lief_binary.header
    timestamp = time.gmtime(fileheader.time_date_stamps)
    return time.strftime('%Y-%m', timestamp)

class PEFeatureExtractor(object):
    ''' Extract useful features from a PE file, and return as a vector of fixed size. '''

    featureslist = [
        ByteHistogram(), ByteEntropyHistogram(), StringExtractor(), GeneralFileInfo(), HeaderFileInfo(), SectionInfo(),
        ImportsInfo(), ExportsInfo()
    ]
    
    # featureslist = [
    #     HeaderFileInfo()
    # ]

    dim = sum([fe.dim for fe in featureslist])

    def raw_features(self, path):
        try:
            lief_binary = lief.PE.parse(path)
            pefile_binary = pefile.PE(path)
            bytez = open(path, 'rb').read()
        except (lief.bad_format, lief.bad_file, lief.pe_error, lief.parser_error, RuntimeError) as e:
            print("lief error: ", str(e))
            lief_binary = None
        except (pefile.PEFormatError) as e :
            print('pefile error ', str(e))
        except Exception:  # everything else (KeyboardInterrupt, SystemExit, ValueError):
            raise

        features = {"appeared" : GenerateTime(lief_binary)} #appeared
        features.update({fe.name: fe.raw_features(bytez, lief_binary, pefile_binary) for fe in self.featureslist})

        return features

    def process_raw_features(self, raw_obj):
        feature_vectors = [fe.process_raw_features(raw_obj[fe.name]) for fe in self.featureslist]
        return np.hstack(feature_vectors).astype(np.float32)

    def feature_vector(self, path):
        return self.process_raw_features(self.raw_features(path))