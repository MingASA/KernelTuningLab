"""Preserve only compiler binaries whose SHA256 is in this study's records.

Reads the existing Triton cache without modifying it. No GPU execution, guessed
binaries, or public Git binary additions. ELF .text bytes are not driver memory.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import struct
import subprocess

import analyze_phase1 as core


def text_sections(blob):
    if len(blob)<64 or blob[:4]!=b'\x7fELF' or blob[4]!=2 or blob[5] not in (1,2): return None
    endian='<' if blob[5]==1 else '>'
    offset=struct.unpack_from(endian+'Q',blob,40)[0]
    size,count,string_index=struct.unpack_from(endian+'HHH',blob,58)
    if size<64 or count==0 or string_index>=count: return None
    def section(index): return struct.unpack_from(endian+'IIQQQQIIQQ',blob,offset+index*size)
    strings=section(string_index);names=blob[strings[4]:strings[4]+strings[5]]
    result=[]
    for index in range(count):
        entry=section(index);end=names.find(b'\0',entry[0]);name=names[entry[0]:end].decode('utf-8','replace')
        if name=='.text' or name.startswith('.text.'):
            value=blob[entry[4]:entry[4]+entry[5]]
            if len(value)!=entry[5]: raise ValueError('invalid ELF text section bounds')
            result.append({'name':name,'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()})
    return result


DEFAULT_STUDIES=('correctness','pilot','matrix','bf16','anchor-before','anchor-after','online','w4','simple','expert-probe')


def archive(root,output,studies_to_scan=DEFAULT_STUDIES,cache_dirs=None):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    wanted={};studies={};wanted_ptx=set()
    for label in studies_to_scan:
        if not (root/label/'summary.json').is_file(): continue
        needed=set()
        for record in core.current_records(root/label):
            resources=record.metadata.get('correctness',{}).get('execution',{}).get('compiled_resources',{})
            for resource in resources.values():
                ptx=resource.get('artifacts',{}).get('ptx')
                if ptx:wanted_ptx.add(ptx['sha256'])
                artifact=resource.get('artifacts',{}).get('cubin')
                if artifact:
                    identity=artifact['sha256'];wanted[identity]=artifact['bytes'];needed.add(identity)
        studies[label]=sorted(needed)
    caches=[Path(p) for p in cache_dirs] if cache_dirs else [Path(os.environ.get('TRITON_CACHE_DIR',Path.home()/'.triton/cache'))]
    filenames=set()
    for cache in caches:
        result=subprocess.run(['rg','--files','--hidden','--no-ignore',str(cache),'-g','*.cubin'],capture_output=True,text=True)
        if result.returncode not in (0,1):raise ValueError('compiler cache scan failed: '+result.stderr.strip())
        filenames.update(result.stdout.splitlines())
    matched={}
    for filename in sorted(filenames):
        path=Path(filename);blob=path.read_bytes();identity=hashlib.sha256(blob).hexdigest()
        if identity not in wanted: continue
        if len(blob)!=wanted[identity]: raise ValueError('recorded compiler binary size mismatch')
        destination=output/(identity+'.cubin')
        if destination.exists() and destination.read_bytes()!=blob: raise ValueError('binary archive collision')
        destination.write_bytes(blob)
        sections=text_sections(blob)
        matched[identity]={'bytes':len(blob),'file':destination.name,'text_sections':sections,
            'text_bytes':sum(s['bytes'] for s in sections) if sections is not None else None}
        ptx_path=path.with_suffix('.ptx')
        if ptx_path.is_file():
            ptx_blob=ptx_path.read_bytes();ptx_identity=hashlib.sha256(ptx_blob).hexdigest()
            if ptx_identity in wanted_ptx:
                ptx_text=ptx_blob.decode('utf-8','replace');ptx_target=re.search(r'(?m)^\s*\.target\s+([^\r\n]+)',ptx_text)
                ptx_file=output/(ptx_identity+'.ptx');ptx_file.write_bytes(ptx_blob)
                matched[identity].update(ptx_sha256=ptx_identity,ptx_file=ptx_file.name,
                    ptx_target=ptx_target.group(1) if ptx_target else None,
                    tensor_core_instructions=sorted(set(re.findall(r'\b(?:mma\.sync[\w.]*|wgmma\.[\w.]+|tcgen05\.[\w.]+)',ptx_text))))
    missing=sorted(set(wanted)-set(matched))
    report={'status':'complete' if not missing else 'partial','wanted_cubin_artifacts':len(wanted),'copied_cubin_artifacts':len(matched),
        'missing_cubin_sha256':missing,'artifacts':matched,'study_cubin_sha256':studies,
        'total_cubin_container_bytes':sum(v['bytes'] for v in matched.values()),
        'known_text_section_bytes':sum(v['text_bytes'] or 0 for v in matched.values()),
        'scope':'Exact SHA256-matching original compiler artifacts. ELF .text section bytes exclude metadata and constants; neither metric is driver-resident size or a solved deployment byte budget.'}
    core.write(output/'index.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('artifacts','study_cubin_sha256')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--studies',nargs='+',default=DEFAULT_STUDIES)
    parser.add_argument('--cache-dirs',nargs='+',help='read-only cache roots, including an isolated cold-study cache')
    args=parser.parse_args();archive(args.root,args.output,args.studies,args.cache_dirs)
