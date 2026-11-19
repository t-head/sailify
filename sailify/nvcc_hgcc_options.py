"""
nvcc → hgcc compiler flag mapping table and translation API.

Each option maps to a dict with keys:
    short        : short-form flag (also a lookup key)
    map_to       : replacement flag(s) — string or list of strings
    needs_ignore : drop the flag (with its value when needs_value)
    needs_value  : the next argument (or the part after '=') is the value
    value_map    : raw value → replacement (or {needs_ignore: True}). A key
                   ending in '=' is a prefix rule on the part before the
                   first '='; the remainder is dropped when it contains a
                   comma (ppu-* targets take a single uint).
    default_value_mapping : policy for unmapped values: 'ignore' drops them;
                   absent → forwarded unchanged.

value_map values follow nvcc's -Xptxas conventions: comma- or space-separated
lists, an operand-taking key paired with a following bare value, and values
concatenated across repeated occurrences. hgcc consumes one value per
occurrence and rejects repeated scalar options, so each translated setting
is emitted once with its final value.
"""

import shlex

NVCC_HGCC_OPTIONS: dict[str, dict] = {
    # ── Direct mappings ──
    '--cubin':                      {'short': '-cubin',                      'map_to': '-hgbin'},
    '--ptx':                        {'short': '-ptx',                         'map_to': '-hgbin'},
    '--cuda':                       {'short': '-cuda',                        'map_to': '-hggc'},
    '--extended-lambda':            {'short': '-extended-lambda',             'map_to': ['--extended-lambda', '-D__CUDACC_EXTENDED_LAMBDA__']},
    '--expt-extended-lambda':       {'short': '-expt-extended-lambda',        'map_to': ['--extended-lambda', '-D__CUDACC_EXTENDED_LAMBDA__']},
    '--nvlink-options':             {'short': '-Xnvlink',                    'map_to': '--hglink-options',            'needs_value': True,
                                   'value_map': {
                                       '--suppress-stack-size-warning': '-ppu-suppress-stack-size-warning',
                                       '-suppress-stack-size-warning': '-ppu-suppress-stack-size-warning',
                                   }},
    '--fatbin-options':             {'short': '-Xfatbin',                     'map_to': '--hgfatbinary-options',       'needs_value': True,
                                   'value_map': {
                                       '--cuda': '--hggc',
                                       '-cuda': '-hggc',
                                   }},
    '--forward-unknown-opts':       {'short': '-forward-unknown-opts',        'map_to': ['--forward-unknown-to-host-linker', '--forward-unknown-to-host-compiler']},
    '--compiler-options':           {'short': '-Xcompiler',                   'map_to': '-Xcompiler',                 'needs_value': True},
    '--preprocessor-options':       {'short': '-Xpreprocessor',               'map_to': '-Xpreprocess',               'needs_value': True},
    '--linker-options':             {'short': '-Xlinker',                     'map_to': '-Xlinker',                   'needs_value': True},
    '--archive-optionss':           {'short': '-Xarchive',                    'map_to': '-Xarchive',                  'needs_value': True},
    '-mllvm':                       {'map_to': '-Xllvm',                                                              'needs_value': True},
    '--allow-unsupported-compiler': {'short': '-allow-unsupported-compiler', 'map_to': ['-Xpreprocess', '-D__NV_NO_HOST_COMPILER_CHECK=1']},
    '--device-debug':               {'short': '-G',                           'map_to': ['-G', '-D__CUDACC_DEBUG__']},
    '--device-w':                   {'short': '-dw',                          'map_to': ['-c', '-rdc=false']},
    '--relocatable-link':           {'short': '-r',                           'map_to': '-r'},
    '--dopt':                       {'short': '-dopt',                        'map_to': '-dopt',                       'needs_value': True,
                                   'value_map': {'off': 0, 'on': 3}},
    # hgcc has no --fsanitize/--apply-controls.
    '--apply-controls':             {'short': '-apply-controls',              'needs_ignore': True,  'needs_value': True},
    '--fdevice-sanitize':           {'short': '-fdevice-sanitize',            'needs_ignore': True,  'needs_value': True},
    '--sanitize':                   {'short': '-sanitize',                    'needs_ignore': True,  'needs_value': True},
    '--time':                       {'short': '-time',                        'map_to': '--time',                     'needs_value': True},
    '--Werror':                     {'short': '-Werror',                       'map_to': '--Werror',                   'needs_value': True,
                                   'value_map': {
                                       'cross-execution-space-call': {'needs_ignore': True},
                                       'default-stream-launch': 'default-stream-launch',
                                       'missing-launch-bounds': 'missing-launch-bounds',
                                       'reorder': 'reorder',
                                       'deprecated-declarations': 'deprecated-declarations',
                                       'ext-lambda-captures-this': 'ext-lambda-captures-this',
                                   }},
    '--ptxas-options':              {'short': '-Xptxas',                      'map_to': '--llvm-options',              'needs_value': True,
                                   'default_value_mapping': 'ignore',
                                   'value_map': {
                                       '-v': '--resource-usage',
                                       '--verbose': '--resource-usage',
                                       '-w': '--no-warn',
                                       '--disable-warnings': '--no-warn',
                                       '--maxntid=': '-ppu-maxntid=',
                                       '-maxntid=': '-ppu-maxntid=',
                                       '--minnctapersm=': '-ppu-minblockscu=',
                                       '-minnctapersm=': '-ppu-minblockscu=',
                                       '--maxrregcount=': '-ppu-max-vreg-count=',
                                       '-maxrregcount=': '-ppu-max-vreg-count=',
                                       # -fatal-warnings: hgcc's opt rejects --warning-as-error
                                       '--warning-as-error': '-fatal-warnings',
                                       '-Werror': '-fatal-warnings',
                                       '-warn-spills': '-ppu-warn-spills',
                                       '--warn-on-spills': '-ppu-warn-spills',
                                       '--warn-on-double-precision-use': '-ppu-warn-double-usage',
                                       '-warn-double-usage': '-ppu-warn-double-usage',
                                       '--warn-on-local-memory-usage': '-ppu-warn-lmem-usage',
                                       '-warn-lmem-usage': '-ppu-warn-lmem-usage',
                                       '-suppress-stack-size-warning': '-ppu-suppress-stack-size-warning',
                                       '--suppress-stack-size-warning': '-ppu-suppress-stack-size-warning',
                                       '--def-load-cache=ca': '-ppu-dlcm=0',
                                       '--def-load-cache=cg': '-ppu-dlcm=2',
                                       '--def-load-cache=cv': '-ppu-dlcm=3',
                                       '--def-load-cache=cs': '-ppu-dlcm=4',
                                       '--def-load-cache=lu': '-ppu-dlcm=5',
                                       '--def-load-cache=wb': '-ppu-dlcm=0',
                                       '--def-load-cache=wt': '-ppu-dlcm=3',
                                       '-dlcm=ca': '-ppu-dlcm=0',
                                       '-dlcm=cg': '-ppu-dlcm=2',
                                       '-dlcm=cv': '-ppu-dlcm=3',
                                       '-dlcm=cs': '-ppu-dlcm=4',
                                       '-dlcm=lu': '-ppu-dlcm=5',
                                       '-dlcm=wb': '-ppu-dlcm=0',
                                       '-dlcm=wt': '-ppu-dlcm=3',
                                       '--def-store-cache=ca': '-ppu-dscm=0',
                                       '--def-store-cache=cg': '-ppu-dscm=2',
                                       '--def-store-cache=cv': '-ppu-dscm=3',
                                       '--def-store-cache=cs': '-ppu-dscm=4',
                                       '--def-store-cache=lu': '-ppu-dscm=5',
                                       '--def-store-cache=wb': '-ppu-dscm=0',
                                       '--def-store-cache=wt': '-ppu-dscm=3',
                                       '-dscm=ca': '-ppu-dscm=0',
                                       '-dscm=cg': '-ppu-dscm=2',
                                       '-dscm=cv': '-ppu-dscm=3',
                                       '-dscm=cs': '-ppu-dscm=4',
                                       '-dscm=lu': '-ppu-dscm=5',
                                       '-dscm=wb': '-ppu-dscm=0',
                                       '-dscm=wt': '-ppu-dscm=3',
                                       '--force-load-cache=ca': '-ppu-flcm=0',
                                       '--force-load-cache=cg': '-ppu-flcm=2',
                                       '--force-load-cache=cv': '-ppu-flcm=3',
                                       '--force-load-cache=cs': '-ppu-flcm=4',
                                       '--force-load-cache=lu': '-ppu-flcm=5',
                                       '--force-load-cache=wb': '-ppu-flcm=0',
                                       '--force-load-cache=wt': '-ppu-flcm=3',
                                       '-flcm=ca': '-ppu-flcm=0',
                                       '-flcm=cg': '-ppu-flcm=2',
                                       '-flcm=cv': '-ppu-flcm=3',
                                       '-flcm=cs': '-ppu-flcm=4',
                                       '-flcm=lu': '-ppu-flcm=5',
                                       '-flcm=wb': '-ppu-flcm=0',
                                       '-flcm=wt': '-ppu-flcm=3',
                                       '--force-store-cache=ca': '-ppu-fscm=0',
                                       '--force-store-cache=cg': '-ppu-fscm=2',
                                       '--force-store-cache=cv': '-ppu-fscm=3',
                                       '--force-store-cache=cs': '-ppu-fscm=4',
                                       '--force-store-cache=lu': '-ppu-fscm=5',
                                       '--force-store-cache=wb': '-ppu-fscm=0',
                                       '--force-store-cache=wt': '-ppu-fscm=3',
                                       '-fscm=ca': '-ppu-fscm=0',
                                       '-fscm=cg': '-ppu-fscm=2',
                                       '-fscm=cv': '-ppu-fscm=3',
                                       '-fscm=cs': '-ppu-fscm=4',
                                       '-fscm=lu': '-ppu-fscm=5',
                                       '-fscm=wb': '-ppu-fscm=0',
                                       '-fscm=wt': '-ppu-fscm=3',
                                       '--position-independent-code': {'needs_ignore': True},
                                       '-pic': {'needs_ignore': True},
                                       '--allow-expensive-optimizations': {'needs_ignore': True},
                                       '-allow-expensive-optimizations': {'needs_ignore': True},
                                       '--register-usage-level': {'needs_ignore': True},
                                       '-register-usage-level': {'needs_ignore': True},
                                       '--disable-optimizer-consts': {'needs_ignore': True},
                                       '-disable-optimizer-consts': {'needs_ignore': True},
                                       '--Ofast-compile': {'needs_ignore': True},
                                       '-Ofc': {'needs_ignore': True},
                                       '--device-stack-protector': {'needs_ignore': True},
                                       '--split-compile': {'needs_ignore': True},
                                       '--sanitize': {'needs_ignore': True},
                                   }},

    # ── Ignored flags (dropped silently) ──
    '--no-display-error-number':           {'short': '-no-err-no',                          'needs_ignore': True},
    '--no-align-double':                                                                   {'needs_ignore': True},
    '--source-in-ptx':                     {'short': '-src-in-ptx',                         'needs_ignore': True},
    '--extra-device-vectorization':        {'short': '-extra-device-vectorization',          'needs_ignore': True},
    '--target-directory':                  {'short': '-target-dir',                         'needs_ignore': True,  'needs_value': True},
    '--diag-suppress':                     {'short': '-diag-suppress',                      'needs_ignore': True,  'needs_value': True},
    '--cudafe-options':                   {'short': '-Xcudafe',                            'needs_ignore': True,  'needs_value': True},
    '--cicc-options':                      {'short': '-Xcicc',                              'needs_ignore': True,  'needs_value': True},
    '--objdir-as-tempdir':                 {'short': '-objtemp',                            'needs_ignore': True},
    '--split-compile':                     {'short': '-split-compile',                      'needs_ignore': True,  'needs_value': True},
    '--Ofast-compile':                     {'short': '-Ofc',                               'needs_ignore': True,  'needs_value': True},
    '--gen-opt-lto':                       {'short': '-gen-opt-lto',                        'needs_ignore': True},
    '--no-host-device-initializer-list':   {'short': '-nohdinitlist',                       'needs_ignore': True},
    '--no-host-device-move-forward':       {'short': '-nohdmoveforward',                    'needs_ignore': True},
    '--device-entity-has-hidden-visibility': {'short': '-device-entity-has-hidden-visibility', 'map_to': '-device-entity-has-hidden-visibility', 'needs_value': True},
    '--dont-use-profile':                  {'short': '-noprof',                             'needs_ignore': True},
    '--split-compile-extended':            {'short': '-split-compile-extended',             'needs_ignore': True,  'needs_value': True},
    '--fdevice-syntax-only':               {'short': '-fdevice-syntax-only',                'needs_ignore': True},
    '--fdevice-time-trace':                {'short': '-fdevice-time-trace',                 'needs_ignore': True,  'needs_value': True},
    '--entries':                           {'short': '-e',                                  'needs_ignore': True,  'needs_value': True},
    '--Wext-lambda-captures-this':         {'short': '-Wext-lambda-captures-this',          'needs_ignore': True},
    '--Wno-deprecated-declarations':       {'short': '-Wno-deprecated-declarations',        'needs_ignore': True},
    '--Wno-deprecated-gpu-targets':        {'short': '-Wno-deprecated-gpu-targets',         'needs_ignore': True},
    '--extensible-whole-program':          {'short': '-ewp',                               'needs_ignore': True},
    '--display-error-number':              {'short': '-err-no',                             'needs_ignore': True},
    '--diag-error':                        {'short': '-diag-error',                         'needs_ignore': True,  'needs_value': True},
    '--diag-warn':                         {'short': '-diag-warn',                          'needs_ignore': True,  'needs_value': True},
    '--host-linker-script':                {'short': '-hls',                                'needs_ignore': True,  'needs_value': True},
    '--augment-host-linker-script':        {'short': '-aug-hls',                            'needs_ignore': True},
    '--brief-diagnostic':                  {'short': '-brief-diag',                         'needs_ignore': True,  'needs_value': True},
    '--jump-table-density':                {'short': '-jtd',                                'needs_ignore': True,  'needs_value': True},
    '--relocatable-ptx':                   {'short': '-reloc-ptx',                          'needs_ignore': True},
    '--device-stack-protector':            {'short': '-device-stack-protector',             'needs_ignore': True,  'needs_value': True},
    '--frandom-seed':                      {'short': '-frandom-seed',                       'needs_ignore': True,  'needs_value': True},
    '--jobserver':                         {'short': '-jobserver',                          'needs_ignore': True},
    '--link':                              {'short': '-link',                               'needs_ignore': True},
    '--optix-ir':                          {'short': '-optix-ir',                           'needs_ignore': True},
    '--suppress-arch-warning':             {'short': '-suppress-arch-warning',              'needs_ignore': True},
    '--cudadevrt':                         {'short': '-cudadevrt',                          'needs_ignore': True,  'needs_value': True},
    '--no-hggc-embed-bc':                  {'short': '-no-hggc-embed-bc',                   'needs_ignore': True},
    '--options-file':                      {'short': '-optf',                               'needs_ignore': True,  'needs_value': True},
    # -gencode re-targets the arch; callers pass --gpu-architecture instead.
    '--generate-code':                     {'short': '-gencode',                            'needs_ignore': True,  'needs_value': True},
}

_NO_RULE = object()
_OPTION_LOOKUP = None


def _option_lookup():
    """Long-name keys plus each 'short' as an extra lookup key."""
    global _OPTION_LOOKUP
    if _OPTION_LOOKUP is None:
        lookup = {}
        for name, entry in NVCC_HGCC_OPTIONS.items():
            lookup[name] = entry
            short = entry.get('short')
            if short and short not in lookup:
                lookup[short] = entry
        _OPTION_LOOKUP = lookup
    return _OPTION_LOOKUP


def _apply_rule(rule):
    if isinstance(rule, dict):
        return None if rule.get('needs_ignore') else rule.get('map_to')
    return str(rule)


def _lookup_value(value_map, value, default_mapping):
    """Exact key first, then bare key (drop rule) or key+'=' (prefix rule)
    matching on the part before the first '='. Returns the translation, None
    to drop, or _NO_RULE when nothing matches."""
    rule = value_map.get(value)
    if rule is not None:
        return _apply_rule(rule)
    head, eq, rest = value.partition('=')
    if eq:
        if head in value_map:
            return _apply_rule(value_map[head])
        rule = value_map.get(head + '=')
        if rule is not None:
            if isinstance(rule, str):
                return None if ',' in rest else rule + rest
            return _apply_rule(rule)
    return _NO_RULE


def _expand_options_files(tokens):
    """Splice --options-file/-optf contents into the token stream, as nvcc
    does before parsing its command line."""
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        head, eq, rest = token.partition('=')
        if token in ('--options-file', '-optf'):
            i += 1
            if i >= n:
                raise ValueError('argument expected after ' + token)
            filename = tokens[i]
            i += 1
        elif eq and head in ('--options-file', '-optf'):
            filename = rest
            i += 1
        else:
            out.append(token)
            i += 1
            continue
        with open(filename) as fh:
            out.extend(shlex.split(fh.read()))
    return out


def _resolve_value(value_map, value, default_mapping):
    """_lookup_value plus the default policy: unknown values drop under
    'ignore' and forward unchanged otherwise."""
    t = _lookup_value(value_map, value, default_mapping)
    if t is _NO_RULE:
        return [] if default_mapping == 'ignore' else [value]
    return [] if t is None else [t]


def _split_elements(value, value_map):
    """Split an option-list value on commas and spaces, keeping an
    operand-taking key with its following bare value and multi-dimension
    operands ('--maxntid=128,4') as one element."""
    elements = []
    for seg in value.replace(',', ' ').split():
        if elements:
            prev = elements[-1]
            head = prev.partition('=')[0]
            if head + '=' in value_map and not seg.startswith('-'):
                elements[-1] = prev + ('=' if '=' not in prev else ',') + seg
                continue
        elements.append(seg)
    return elements


def _translate_values(entry, values):
    """Resolve the values from every occurrence of *entry* (nvcc concatenates
    them). hgcc rejects repeated scalar options, so each translated option is
    emitted once with its final value."""
    value_map = entry['value_map']
    default_mapping = entry.get('default_value_mapping')
    combined = ' '.join(values)
    # exact whole-value match wins over list splitting
    if value_map.get(combined) is not None:
        elements = [combined]
    else:
        elements = _split_elements(combined, value_map)
    kept = {}
    for elem in elements:
        for t in _resolve_value(value_map, elem, default_mapping):
            kept[t.partition('=')[0]] = t
    return list(kept.values())


def translate_nvcc_options(options):
    """Translate an nvcc compile-option string (or argv list) to hgcc args.

    Long/short spellings and joined '=' forms both work; --options-file
    contents are spliced in first. value_map values are concatenated across
    occurrences, split into elements, and emitted as one '<map_to> <value>'
    pair per element. Options without a value_map are forwarded raw;
    needs_ignore options drop with their value; unmapped options pass
    through unchanged.

    Returns the hgcc argument list.
    """
    if isinstance(options, str):
        tokens = shlex.split(options)
    else:
        tokens = list(options)
    tokens = _expand_options_files(tokens)

    lookup = _option_lookup()
    out = []
    collected = []  # (entry, mapped) in first-seen order
    values_by_entry = {}
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        flag = token
        joined_value = None
        if '=' in token:
            head, _, rest = token.partition('=')
            if head in lookup:
                flag = head
                joined_value = rest
        entry = lookup.get(flag)
        if entry is None:
            out.append(token)
            i += 1
            continue

        if entry.get('needs_ignore'):
            i += 1
            if entry.get('needs_value') and joined_value is None and i < n:
                i += 1
            continue

        map_to = entry.get('map_to')
        if map_to is None:
            i += 1
            continue
        mapped = map_to if isinstance(map_to, list) else [map_to]

        if not entry.get('needs_value'):
            out.extend(mapped)
            i += 1
            continue

        if joined_value is not None:
            value = joined_value
            i += 1
        else:
            i += 1
            if i >= n:
                break
            value = tokens[i]
            i += 1

        if entry.get('value_map'):
            # a bare operand-taking key pairs with the value that follows it
            if '=' not in value and value + '=' in entry['value_map'] \
                    and i < n and not tokens[i].startswith('-'):
                value += ' ' + tokens[i]
                i += 1
            key = id(entry)
            if key not in values_by_entry:
                values_by_entry[key] = []
                collected.append((entry, mapped))
            values_by_entry[key].append(value)
        else:
            out.extend(mapped)
            out.append(value)

    for entry, mapped in collected:
        for t in _translate_values(entry, values_by_entry[id(entry)]):
            out.extend(mapped)
            out.append(t)
    return out
