import sys
from pathlib import Path

import flatbuffers
from ai_edge_litert import schema_py_generated as schema


def convert(source_path, target_path):
    source = source_path.read_bytes()
    model = schema.Model.GetRootAsModel(source, 0)
    model_object = schema.ModelT.InitFromObj(model)

    converted_buffers = 0
    for index, target_buffer in enumerate(model_object.buffers):
        source_buffer = model.Buffers(index)
        offset = source_buffer.Offset()
        size = source_buffer.Size()
        if not offset or not size:
            continue
        end = offset + size
        if end > len(source):
            raise ValueError(f"Buffer {index} fora do arquivo: {offset}:{end}")
        target_buffer.data = bytearray(source[offset:end])
        target_buffer.offset = 0
        target_buffer.size = 0
        converted_buffers += 1

    builder = flatbuffers.Builder(len(source))
    model_offset = model_object.Pack(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    target_path.write_bytes(builder.Output())

    print(f"buffers convertidos: {converted_buffers}")
    print(f"arquivo gerado: {target_path}")
    print(f"tamanho: {target_path.stat().st_size}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"Uso: {sys.argv[0]} ORIGEM DESTINO")

    source_path = Path(sys.argv[1])
    target_path = Path(sys.argv[2])
    if source_path.resolve() == target_path.resolve():
        raise ValueError("Origem e destino devem ser diferentes")

    convert(source_path, target_path)


if __name__ == "__main__":
    main()
