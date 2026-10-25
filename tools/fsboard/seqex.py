"""tf.train.SequenceExample reader/writer WITHOUT TensorFlow.

FSboard stores each clip as a serialized tf.train.SequenceExample.  The message
layout (field numbers) is taken from tensorflow/core/example/{feature,example}.proto
and built at runtime with the `protobuf` package, so TensorFlow is not needed.
"""
from google.protobuf import descriptor_pb2 as d, descriptor_pool, message_factory

_F = d.FieldDescriptorProto


def _build():
    fdp = d.FileDescriptorProto(name="fsb_example.proto", package="fsb", syntax="proto3")

    def msg(name):
        m = fdp.message_type.add(); m.name = name; return m

    def fld(m, name, num, typ, label=_F.LABEL_OPTIONAL, type_name=None, oneof=None):
        f = m.field.add(); f.name = name; f.number = num; f.type = typ; f.label = label
        if type_name: f.type_name = type_name
        if oneof is not None: f.oneof_index = oneof
        return f

    bl = msg("BytesList"); fld(bl, "value", 1, _F.TYPE_BYTES, _F.LABEL_REPEATED)
    fl = msg("FloatList"); fld(fl, "value", 1, _F.TYPE_FLOAT, _F.LABEL_REPEATED)
    il = msg("Int64List"); fld(il, "value", 1, _F.TYPE_INT64, _F.LABEL_REPEATED)
    ft = msg("Feature"); ft.oneof_decl.add().name = "kind"
    fld(ft, "bytes_list", 1, _F.TYPE_MESSAGE, type_name=".fsb.BytesList", oneof=0)
    fld(ft, "float_list", 2, _F.TYPE_MESSAGE, type_name=".fsb.FloatList", oneof=0)
    fld(ft, "int64_list", 3, _F.TYPE_MESSAGE, type_name=".fsb.Int64List", oneof=0)

    fs = msg("Features")
    e = fs.nested_type.add(); e.name = "FeatureEntry"; e.options.map_entry = True
    fld(e, "key", 1, _F.TYPE_STRING); fld(e, "value", 2, _F.TYPE_MESSAGE, type_name=".fsb.Feature")
    fld(fs, "feature", 1, _F.TYPE_MESSAGE, _F.LABEL_REPEATED, ".fsb.Features.FeatureEntry")

    fli = msg("FeatureList"); fld(fli, "feature", 1, _F.TYPE_MESSAGE, _F.LABEL_REPEATED, ".fsb.Feature")
    fls = msg("FeatureLists")
    e = fls.nested_type.add(); e.name = "FeatureListEntry"; e.options.map_entry = True
    fld(e, "key", 1, _F.TYPE_STRING); fld(e, "value", 2, _F.TYPE_MESSAGE, type_name=".fsb.FeatureList")
    fld(fls, "feature_list", 1, _F.TYPE_MESSAGE, _F.LABEL_REPEATED, ".fsb.FeatureLists.FeatureListEntry")

    se = msg("SequenceExample")
    fld(se, "context", 1, _F.TYPE_MESSAGE, type_name=".fsb.Features")
    fld(se, "feature_lists", 2, _F.TYPE_MESSAGE, type_name=".fsb.FeatureLists")

    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    return {n: message_factory.GetMessageClass(pool.FindMessageTypeByName("fsb." + n))
            for n in ("SequenceExample", "Feature", "FeatureList")}


_C = _build()
SequenceExample = _C["SequenceExample"]


def parse(serialized: bytes):
    """-> (context: dict[str, value], lists: dict[str, list[list[float]]])"""
    ex = SequenceExample(); ex.ParseFromString(serialized)
    ctx = {}
    for k, v in ex.context.feature.items():
        kind = v.WhichOneof("kind")
        if kind == "bytes_list": ctx[k] = v.bytes_list.value[0].decode("utf-8", "replace") if v.bytes_list.value else ""
        elif kind == "int64_list": ctx[k] = int(v.int64_list.value[0]) if v.int64_list.value else 0
        elif kind == "float_list": ctx[k] = float(v.float_list.value[0]) if v.float_list.value else 0.0
    lists = {}
    for k, fl in ex.feature_lists.feature_list.items():
        rows = []
        for f in fl.feature:
            kind = f.WhichOneof("kind")
            rows.append(list(getattr(f, kind).value) if kind else [])
        lists[k] = rows
    return ctx, lists


def build(video_name: str, prompt: str, hands: dict, num_frames: int) -> bytes:
    """Make a SequenceExample (used by tests/synthetic data).
    hands: {"x_right_hand": [[21 floats] or [] per frame], ...}"""
    ex = SequenceExample()
    ex.context.feature["video_name"].bytes_list.value.append(video_name.encode())
    ex.context.feature["prompt"].bytes_list.value.append(prompt.encode())
    ex.context.feature["num_frames"].int64_list.value.append(num_frames)
    for k, rows in hands.items():
        fl = ex.feature_lists.feature_list[k]
        for r in rows:
            f = fl.feature.add()
            if r: f.float_list.value.extend(r)
    return ex.SerializeToString()
