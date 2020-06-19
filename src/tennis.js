/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */
/* TenniS: Tensor based Edge Neural Network Inference System */

var tennis = tennis || {};
var base = base || require('./base');
var long = long || { Long: require('long') };
var marked = marked || require('marked');
var utils = utils || {};

utils.Stream = class {
    /**
     * 
     * @param {DataView|ArrayBuffer} data 
     * @param {number} offset
     */
    constructor(data, offset=0) {
        var Buffer = Buffer || class {};
        var SharedArrayBuffer = SharedArrayBuffer || class {};
        if (data instanceof DataView) {
            this._dataview = data;
            this._offset = offset;
        } else if (data instanceof Buffer) {
            this._dataview = new DataView(data.buffer, data.byteOffset, data.length);
            this._offset = offset;
        } else if (data instanceof ArrayBuffer ||
                data instanceof SharedArrayBuffer) {
            this._dataview = new DataView(data);
            this._offset = offset;
        } else if (data instanceof Uint8Array) {
            this._dataview = new DataView(data.buffer, data.byteOffset, data.length);
            this._offset = offset;
        } else {
            throw utils.Error("Stream param 1 must be DataView or ArrayBuffer");
        }
    }

    /**
     * 
     */
    rewind() {
        this._offset = 0;
    }

    /**
     * @return {number} 
     */
    int64() {
        let high = this.uint32();
        let low = this.uint32();
        return new long.Long(high, low, false).toNumber();
    }

    /**
     * @return {number} 
     */
    uint64() {
        let high = this.uint32();
        let low = this.uint32();
        return new long.Long(high, low, true).toNumber();
    }

    float32() {
        const value = this._dataview.getFloat32(this._offset, true);
        this._offset += 4;
        return value;
    }

    float64() {
        const value = this._dataview.getFloat64(this._offset, true);
        this._offset += 8;
        return value;
    }

    /**
     * @return {number}
     */
    offset() {
        return this._offset;
    }

    /**
     * @return {number}
     */
    int8() {
        const value = this._dataview.getInt8(this._offset, true);
        this._offset += 1;
        return value;
    }

    /**
     * @return {number}
     */
    int16() {
        const value = this._dataview.getInt16(this._offset, true);
        this._offset += 2;
        return value;
    }

    /**
     * @return {number}
     */
    int32() {
        const value = this._dataview.getInt32(this._offset, true);
        this._offset += 4;
        return value;
    }

    /**
     * @return {number}
     */
    uint8() {
        const value = this._dataview.getUint8(this._offset, true);
        this._offset += 1;
        return value;
    }

    /**
     * @return {number}
     */
    uint16() {
        const value = this._dataview.getUint16(this._offset, true);
        this._offset += 2;
        return value;
    }

    /**
     * @return {number}
     */
    uint32() {
        const value = this._dataview.getUint32(this._offset, true);
        this._offset += 4;
        return value;
    }

    /**
     * 
     * @param {number} size 
     * @return {ArrayBuffer}
     */
    buffer(size) {
        let data = this._dataview;
        const beg = data.byteOffset + this._offset;
        const end = beg + size;
        this._offset += size;
        return this._dataview.buffer.slice(beg, end);
    }

    /**
     * @param {number} size
     */
    skip(size) {
        this._offset += size;
    }

    /**
     * @return {string}
     */
    string() {
        const length = this.int32();
        const subbuffer = this.buffer(length);
        let name = String.fromCharCode.apply(null, new Int8Array(subbuffer));
        return name;
        // var buffer_blob = new Blob([subbuffer]);
        // var buffer_reader = new FileReader();
        // buffer_reader.readAsText(buffer_blob, 'utf-8');
        // return buffer_reader.result;
    }

    /**
     * @return {[number]}
     */
    int32_array() {
        const size = this.int32();
        let array = [];
        for (let i = 0; i < size; ++i) {
            array.push(this.int32());
        }
        return array;
    }
};

utils.dtype = {
    VOID: 0,
    INT8: 1,
    UINT8: 2,
    INT16: 3,
    UINT16: 4,
    INT32: 5,
    UINT32: 6,
    INT64: 7,
    UINT64: 8,
    FLOAT16: 9,
    FLOAT32: 10,
    FLOAT64: 11,
    PTR: 12,
    CHAR8: 13,
    CHAR16: 14,
    CHAR32: 15,
    UNKNOWN8: 16,
    UNKNOWN16: 17,
    UNKNOWN32: 18,
    UNKNOWN64: 19,
    UNKNOWN128: 20,
    BOOLEAN: 21,
    COMPLEX32: 22,
    COMPLEX64: 23,
    COMPLEX128: 24,

    /**
     * 
     * @param {number} dtype
     * @return {number} 
     */
    type_bytes: function(dtype) {
        switch (dtype) {
            case this.VOID: return 0;
            case this.INT8: return 1;
            case this.UINT8: return 1;
            case this.INT16: return 2;
            case this.UINT16: return 2;
            case this.INT32: return 4;
            case this.UINT32: return 4;
            case this.INT64: return 8;
            case this.UINT64: return 8;
            case this.FLOAT16: return 2;
            case this.FLOAT32: return 4;
            case this.FLOAT64: return 8;
            case this.CHAR8: return 1;
            case this.CHAR16: return 2;
            case this.CHAR32: return 4;
            case this.UNKNOWN8: return 1;
            case this.UNKNOWN16: return 2;
            case this.UNKNOWN32: return 4;
            case this.UNKNOWN64: return 8;
            case this.UNKNOWN128: return 16;
            case this.BOOLEAN: return 1;
            case this.COMPLEX32: return 4;
            case this.COMPLEX64: return 8;
            case this.COMPLEX128: return 16;
            default:
                break;
        }
        throw utils.Error("Not support dtype = " + dtype);
    },

    /**
     * 
     * @param {number} dtype
     * @return {string} 
     */
    type_str: function(dtype) {
        switch (dtype) {
            case this.VOID: return "void";
            case this.INT8: return "int8";
            case this.UINT8: return "uint8";
            case this.INT16: return "int16";
            case this.UINT16: return "uint16";
            case this.INT32: return "int32";
            case this.UINT32: return "uint32";
            case this.INT64: return "int64";
            case this.UINT64: return "uint64";
            case this.FLOAT16: return "float16";
            case this.FLOAT32: return "float32";
            case this.FLOAT64: return "float64";
            case this.CHAR8: return "char8";
            case this.CHAR16: return "char16";
            case this.CHAR32: return "char32";
            case this.UNKNOWN8: return "unknown8";
            case this.UNKNOWN16: return "unknown16";
            case this.UNKNOWN32: return "unknown32";
            case this.UNKNOWN64: return "unknown64";
            case this.UNKNOWN128: return "unknown128";
            case this.BOOLEAN: return "bool";
            case this.COMPLEX32: return "complex32";
            case this.COMPLEX64: return "complex64";
            case this.COMPLEX128: return "complex128";
            default: return "unkown";
        }
    },
};

utils.Tensor = class {
    /**
     * 
     * @param {[number]} shape 
     * @param {number} dtype 
     * @param {ArrayBuffer} data 
     * @param {object} value 
     */
    constructor(shape, dtype, data, value=null) {
        this._shape = shape;
        this._dtype = dtype;
        this._data = data; // can be null
        this._field = [];
        
        if (value) {
            this._value = value;
        } else {
            this._value = null;  // for js readable value
        }
    }

    get shape() {
        return this._shape;
    }

    get dtype() {
        return this._dtype;
    }

    get count() {
        let prod = 1;
        for (const i of this._shape) {
            prod *= i;
        }
        return prod;
    }

    get proto() {
        return new utils.Prototype(this._dtype, this._shape);
    }
    /**
     * 
     * @param {Function} decoder 
     */
    _decode_core(decoder) {
        if (this._shape.length == 0) {
            return decoder();
        }
        const count = this.count;
        let list = [];
        for (let i = 0; i < count; ++i) {
            list.push(decoder());
        }
        return list;
    }

    _decode() {
        let stream = new utils.Stream(this._data);
        switch (this._dtype) {
        case utils.dtype.INT8:
            return this._decode_core(function() {return stream.int8(); });
        case utils.dtype.INT16:
            return this._decode_core(function() {return stream.int16(); });
        case utils.dtype.INT32:
            return this._decode_core(function() {return stream.int32(); });
        case utils.dtype.INT64:
            return this._decode_core(function() {return stream.int64(); });
        case utils.dtype.UINT8:
            return this._decode_core(function() {return stream.uint8(); });
        case utils.dtype.UINT16:
            return this._decode_core(function() {return stream.uint16(); });
        case utils.dtype.UINT32:
            return this._decode_core(function() {return stream.uint32(); });
        case utils.dtype.UINT64:
            return this._decode_core(function() {return stream.uinst64(); });
        case utils.dtype.FLOAT32:
            return this._decode_core(function() {return stream.float32(); });
        case utils.dtype.FLOAT64:
            return this._decode_core(function() {return stream.float64(); });
        case utils.dtype.BOOLEAN:
            return this._decode_core(function() {return stream.int8() ? true : false; });
        }
        return null;
    }

    get value() {
        if (!(this._value === null)) {
            return this._value;
        }
        switch (this._dtype) {
        case utils.dtype.CHAR8:
            {
                this._value = String.fromCharCode.apply(null, new Int8Array(this._data));
                break;
            }
        default:
            {
                this._value = this._decode();
            }
        }
        return this._value;
    }

    _context() {
        let context = {};
        context.stream = new utils.Stream(this._data);
        let decoder = null;
        switch (this._dtype) {
            case utils.dtype.INT8:
                decoder = function(stream) {return stream.int8(); }; break;
            case utils.dtype.INT16:
                decoder = function(stream) {return stream.int16(); }; break;
            case utils.dtype.INT32:
                decoder = function(stream) {return stream.int32(); }; break;
            case utils.dtype.INT64:
                decoder = function(stream) {return stream.int64(); }; break;
            case utils.dtype.UINT8:
                decoder = function(stream) {return stream.uint8(); }; break;
            case utils.dtype.UINT16:
                decoder = function(stream) {return stream.uint16(); }; break;
            case utils.dtype.UINT32:
                decoder = function(stream) {return stream.uint32(); }; break;
            case utils.dtype.UINT64:
                decoder = function(stream) {return stream.uint64(); }; break;
            case utils.dtype.FLOAT32:
                decoder = function(stream) {return stream.float32(); }; break;
            case utils.dtype.FLOAT64:
                decoder = function(stream) {return stream.float64(); }; break;
            case utils.dtype.BOOLEAN:
                decoder = function(stream) {return stream.uint8(); }; break;
            }
        if (decoder === null) {
            return null;
        }
        context.shape = this._shape;
        context.next = decoder;
        context.count = 0;
        context.limit = Number.MAX_SAFE_INTEGER;

        return context;
    }

    get viewable() {
        return (false ||
            this._dtype == utils.dtype.INT8 ||
            this._dtype == utils.dtype.INT16 ||
            this._dtype == utils.dtype.INT32 ||
            this._dtype == utils.dtype.INT64 ||
            this._dtype == utils.dtype.UINT8 ||
            this._dtype == utils.dtype.UINT16 ||
            this._dtype == utils.dtype.UINT32 ||
            this._dtype == utils.dtype.UINT64 ||
            this._dtype == utils.dtype.FLOAT32 ||
            this._dtype == utils.dtype.FLOAT64 ||
            this._dtype == utils.dtype.BOOLEAN);
    }

    _view(context, dim=0) {
        if (dim >= context.shape.length) {
            return context.next(context.stream);
        }
        let array = [];
        const size = context.shape[dim];
        if (dim == context.shape.length - 1) {
            for (let i = 0; i < size; ++i) {
                if (context.count >= context.limit) {
                    array.push("...");
                    return array;
                }
                array.push(context.next(context.stream));
                context.count++;
            }
        } else {
            for (let i = 0; i < size; ++i) {
                if (context.count >= context.limit) {
                    array.push("...");
                    return array;
                }
                array.push(this._view(context, dim + 1));
            }
        }
        return array;
    }

    view(limit=null) {
        let context = this._context();
        if (context === null) return null;
        if (!(limit === null)) {
            context.limit = limit;
        }
        return this._view(context);
    }

    /**
     * @param {[utils.Tensor]} fields
     * @return {utils.Tensor}
     */
    static Pack(fields) {
        if (fields.length == 0) {
            new utils.Tensor([], utils.dtype.VOID, null);
        } else if (fields.length == 1) {
            return fields[0];
        }
        let extra = [];
        for (let i = 1; i < fields.length; ++i) {
            extra.push(fields[i]);
        }
        let packed = new utils.Tensor(this._shape, this._dtype, this._data, this._value);
        packed._field = extra;
        return packed;
    }
};

utils.Node = class {
    /**
     * 
     * @param {{}} params 
     */
    constructor(params, id=null) {
        // for (let param in params) {
        //     // no use code
        //     let value = params[param];
        // }
        this._id = id;
        this._params = params;
        this._name = this.get("#name").value;
        this._op = this.get("#op").value;
        this._shape = this.get("#shape");
        this._dtype = this.get("#dtype");
        this._inputs = [];
        this._outputs = [this];
        this._hand_arg_id = null;

        if (!(this._dtype === null)) {
            this._dtype = this._dtype.value;
        }

        if (!(this._shape === null)) {
            this._shape = this._shape.value;
        }
    }

    get proto() {
        if (this._dtype === null && this._shape === null) {
            return null;
        }
        return new utils.Prototype(this._dtype, this._shape);
    }

    /**
     * @return {string} arg_id
     */
    get arg_id() {
        if (this._hand_arg_id) {
            return this._hand_arg_id;
        }
        return "[" + this._id.toString() + "] " + this._name;
    }

    set arg_id(v) {
        this._hand_arg_id = v;
    }

    /**
     * @return {number} id
     */
    get id() {
        return this._id;
    }

    /**
     * @param {number} v
     */
    set id(v) {
        this._id = v;
    }

    /**
     * @return {string} name
     */
    get name() { return this._name; }

    /**
     * @return {string} op
     */
    get op() { return this._op; }

    /**
     * @return {[number]} shape
     */
    get shape() { return this._shape; }

    /**
     * @return {number} dtype
     */
    get dtype() { return this._dtype; }

    /**
     * 
     * @return {{}} params
     */
    get params() { return this._params; }

    /**
     * 
     * @param {name} param 
     */
    has(param) {
        return param in this._params;
    }

    /**
     * 
     * @param {string} param 
     * @param {*} value 
     */
    get(param, value=null) {
        if (param in this._params) {
            return this._params[param];
        }
        return value;
    }

    /**
     * @return {[utils.Node]}
     */
    get inputs() {
        return this._inputs;
    }

    /**
     * @param {[utils.Node]} nodes 
     */
    set inputs(v) {
        this._inputs = v;
    }

    /**
     * @return {[utils.Node]}
     */
    get outputs() {
        return this._outputs;
    }

    /**
     * 
     * @param {number} i 
     * @return {utils.Node}
     */
    input(i) {
        return this._inputs[i];
    }
    /**
     * 
     * @param {number} i 
     * @return {utils.Node}
     */
    output(i) {
        return this._outputs[i];
    }
};

utils.Prototype =  class {
    /**
     * 
     * @param {number} dtype 
     * @param {[number]} shape 
     */
    constructor(dtype, shape) {
        this._dtype = dtype;
        this._shape = shape;
    }

    toString() {
        let part1 = this._dtype === null ? "" : utils.dtype.type_str(this._dtype);
        let part2 = "";
        if (this._shape === null) {
            part2 = part1.length > 0 ? "[...]" : "tensor";
        } else if (this._shape.length > 0) {
            part2 += "[";
            for (let i = 0; i < this._shape.length; ++i) {
                if (i > 0) part2 += ", ";
                let dim = this._shape[i];
                if (dim < 0) {
                    part2 += "?";
                } else {
                    part2 += dim;
                }
            }
            part2 += "]";
        }
        return part1 + part2;
    }
};

utils.Module = class {
    /**
     * 
     * @param {utils.Stream} stream 
     */
    constructor(stream) {
        stream.skip(4);
        let mask = stream.int32();
        if (mask != 0x19910929) {
            throw utils.Error("TenniS Module not valid.");
        }
        this._mask = mask;
        stream.skip(120);
        let inputs = stream.int32_array();
        let outputs = stream.int32_array();
        this._nodes = [];
        this._read_graph(stream);
        this._inputs = [];
        this._outputs = [];
        for (const i of inputs) {
            this._inputs.push(this._nodes[i]);
        }
        for (const i of outputs) {
            this._outputs.push(this._nodes[i]);
        }
    }

    /**
     * @return {number}
     */
    get mask() { return this._mask; }

    /**
     * @return {[utils.Node]} inputs
     */
    get inputs() { return this._inputs; }

    /**
     * @return {[utils.Node]} outputs
     */
    get outputs() { return this._outputs; }

    /**
     * @return {[utils.Node]} nodes
     */
    get nodes() { return this._nodes; }

    /**
     * @param {[number]} shape
     * @return {number}
     */
    _prod(shape) {
        let prod = 1;
        for (const i of shape) {
            prod *= i;
        }
        return prod;
    }

    /**
     * 
     * @param {utils.Stream} stream
     * @return {utils.Tensor} 
     */
    _read_tensor(stream) {
        let size = stream.int32();
        let fileds = [];
        for (let i = 0; i < size; ++i) {
            let dtype = stream.int8();
            let shape = stream.int32_array();
            let data_size = utils.dtype.type_bytes(dtype) * this._prod(shape);
            let data = stream.buffer(data_size);
            let tensor = new utils.Tensor(shape, dtype, data);
            fileds.push(tensor);
        }
        return utils.Tensor.Pack(fileds);
    }

    /**
     * 
     * @param {utils.Stream} stream
     * @param {number} id
     * @return {utils.Node} 
     */
    _read_bubble(stream, id=null) {
        let size = stream.int32();
        let params = {};
        for (let i = 0; i < size; ++i) {
            let name = stream.string();
            let value = this._read_tensor(stream);
            params[name] = value;
        }
        return new utils.Node(params, id);
    }

    /**
     * 
     * @param {utils.Stream} stream 
     */
    _read_graph(stream) {
        let size = stream.int32();
        let nodes = [];
        let node_inputs = [];
        for (let i = 0; i < size; ++i) {
            nodes.push(this._read_bubble(stream, i));
            node_inputs.push(stream.int32_array());
        }
        for (let i = 0; i < size; ++i) {
            let inputs = [];
            for (const j of node_inputs[i]) {
                inputs.push(nodes[j]);
            }
            nodes[i].inputs = inputs;
        }
        this._nodes = nodes;
    }
};

tennis.ModelFactory = class {

    match(context) {
        // const extension = context.identifier.split('.').pop().toLowerCase();
        const b = context.buffer;
        let stream = new utils.Stream(b);
        stream.skip(4);
        const mask = stream.int32();
        return mask == 0x19910929;
    }

    open(context, host) {
        return tennis.Metadata.open(host).then((metadata) => {
            const identifier = context.identifier;
            let stream = new utils.Stream(context.buffer);
            return this._openModel(metadata, stream, identifier);
        });
    }

    _openModel(metadata, stream, identifier) {
        try {
            return new tennis.Model(metadata, stream);
        }
        catch (error) {
            const message = error && error.message ? error.message : error.toString();
            // message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
            throw new tennis.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
        }
    }
};

tennis.Model = class {

    constructor(metadata, stream) {
        this._graphs = [];
        this._metadata = [];

        let graph = new tennis.Graph(metadata, stream);
        this._graphs.push(graph);

        this._metadata.push({ name: "version", value: "0x" + graph.mask.toString(16)});
    }

    /**
     * @return {number}
     */
    get metadata() { return this._metadata; }

    get format() {
        return 'tennis';
    }

    get graphs() {
        return this._graphs;
    }
};

tennis.Graph = class {
    
    constructor(metadata, stream) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        let graph = new utils.Module(stream);
        this._mask = graph.mask;

        for (const input of graph.inputs) {
            this._inputs.push(new tennis.Parameter(input.name, true, [new tennis.Argument(input, input.proto, "", null)]));
        }
        for (const output of graph.outputs) {
            this._outputs.push(new tennis.Parameter(output.name, true, [new tennis.Argument(output, output.proto, "", null)]));
        }

        const node_v2_map = {
            "conv2d_v2": 1,
            "depthwise_conv2d_v2": 1,
            "pooling2d_v2": 1,
        };

        const node_copy_map = {
            "_copy": 0,
        };

        let node_output_count = {};
        for (let node of graph.nodes) {
            node_output_count[node] = 0;
            for (let input of node.inputs) {
                node_output_count[input] += 1;
            }
        }

        let draw_nodes = [];
        for (let node of graph.nodes) {
            if (node.op == "<param>") continue;
            if (node.op == "<const>") continue;

            if (node.op in node_v2_map) {
                let input = node.input(node_v2_map[node.op]);
                input.chain = input.chain || [];
                input.chain.push(node);
                continue;
            }

            if (node.op in node_copy_map) {
                let input = node.input(node_copy_map[node.op]);
                if (node_output_count[input] == 1) {
                    while (input.op in node_copy_map) {
                        let tmp = input.input(node_copy_map[input.op]);
                        if (node_output_count[tmp] != 1) {
                            break;
                        }
                        input = tmp;
                    }
                    input.chain = input.chain || [];
                    input.chain.push(node);
                    continue;
                }
            }

            draw_nodes.push(node);
        }

        for (const node of draw_nodes) {
            let draw_node = new tennis.Node(metadata, node);
            if (node.chain && node.chain.length > 0) {
                draw_node.chain = draw_node.chain || [];
                for (const chain of node.chain) {
                    draw_node.chain.push(new tennis.Node(metadata, chain));
                }
            }
            this._nodes.push(draw_node);
        }
    }

    /**
     * @return {number}
     */
    get mask() { return this._mask; }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }
};

tennis.Parameter = class {

    constructor(name, visible, args) {
        this._name = name;
        this._visible = visible;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get arguments() {
        return this._arguments;
    }
};

tennis.Argument = class {
    /**
     * 
     * @param {utils.Node} node 
     * @param {object} type 
     * @param {string} description 
     * @param {object} initializer 
     */
    constructor(node, type, description, initializer) {
        this._id = node.arg_id;
        let proto = node.proto;
        if (proto) {
            this._type = proto;
        } else {
            this._type = type || null;
        }
        this._description = description || '';
        this._initializer = initializer || null;
        if (!this._initializer) {
            if (node.op == "<const>") {
                this._initializer = new tennis.Tensor(node.get("value"), node.name);
            }
        }
        this._name = node.name;
    }

    /// [Deprecated]
    get id() {
        return this._id;
    }

    /// [New in v4]
    get name() {
        return this._id;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    get description() {
        return this._description;
    }

    set type(value) {
        if (this._type) {
            throw new tennis.Error('Invalid argument type set operation.');
        }
        this._type = value;
    }

    get initializer() {
        return this._initializer;
    }
};

tennis.Node = class {
    /**
     * 
     * @param {tennis.Metadata} metadata 
     * @param {utils.Node} node 
     */
    constructor(metadata, node) {
        this._name = node.name;
        this._metadata = metadata;
        this._operator = node.op;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._chain = [];

        /**
         * add input output info
         */
        let schema = this._metadata.getSchema(this._operator)
        let schema_inputs = [];
        let schema_outputs = [];
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema_inputs = schema.inputs;
            schema_outputs = schema.outputs;
        } else {
            for (let i = 0; i < node.inputs.length; ++i) {
                schema_inputs.push({
                    name: "" + i,
                    type: "tensor",
                    description: "",
                });
            }
            schema_outputs.push({
                name: "" + 0,
                type: "tensor",
                description: "",
            });
        }
        // set inputs which have schema
        let i = null;   // for schema index
        let ii = null;  // for input index
        for (i = ii = 0; i < schema_inputs.length; ++i, ++ii) {
            if (ii >= node.inputs.length) break;
            const input = schema_inputs[i];
            if (input.option == "variadic") {
                let left_schema = schema_inputs.length - i;
                let left_input = node.inputs.length - ii;
                let variadic_count = left_input - left_schema + 1;
                let args = node.inputs.slice(ii, ii + variadic_count).map(v => {
                    return new tennis.Argument(v, null, null, null);
                });
                this._inputs.push(new tennis.Parameter(input.name, true, args));
                ii += variadic_count - 1;
                continue;
            }
            this._inputs.push(new tennis.Parameter(input.name, true, [
                new tennis.Argument(node.input(ii), input.type, input.description)
            ]));
        }
        // set inputs with out schema
        for (i = ii; i < node.inputs.length; ++i) {
            let input = {name: "" + i, type: "tensor", description: ""};
            this._inputs.push(new tennis.Parameter(input.name, true, [
                new tennis.Argument(node.input(i), input.type, input.description)
            ]));
        }
        // set output with schema
        for (i = 0; i < node.outputs.length; ++i) {
            if (i >= schema_outputs.length) break;
            const output = schema_outputs[i];
            // do not check option, for node.outputs.length always be 1
            this._outputs.push(new tennis.Parameter(output.name, true, [
                new tennis.Argument(node.output(i), output.type, output.description)
            ]));
        }
        // set output without schema
        for (; i < node.outputs.length; ++i) {
            let output = {name: "" + i, type: "tensor", description: ""};
            this._outputs.push(new tennis.Parameter(output.name, true, [
                new tennis.Argument(node.output(i), output.type, output.description)
            ]));
        }
        // set output only schema
        for (; i < schema_outputs.length; ++i) {
            let output = schema_outputs[i];
            let node_output = {arg_id: "null", proto: output.type, op: "<fake>"}; // for inner used
            this._outputs.push(new tennis.Parameter(output.name, true, [
                new tennis.Argument(node_output, output.type, output.description)
            ]));
        }

        /**
         * TODO: Show the dtype, shape, value for each node frozen debug
         */

        /**
         * add attrubite
         */
        for (let name in node.params) {
            if (name[0] == '#') continue;
            let value = node.get(name);
            this._attributes.push(new tennis.Attribute(metadata, node.op, name, value));
        }
    }

    get name() {
        return this._name;
    }

    /// [Deprecated]
    get operator() {
        return this._operator;
    }

    get type() {
        return this._operator;
    }

    /// [Deprecated]
    get documentation() {
        let schema = this._metadata.getSchema(this._operator);
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = this._operator;
            if (schema.description) {
                schema.description = marked(schema.description);
            }
            if (schema.attributes) {
                for (const attribute of schema.attributes) {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description);
                    }
                }
            }
            if (schema.inputs) {
                for (const input of schema.inputs) {
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                }
            }
            if (schema.outputs) {
                for (const output of schema.outputs) {
                    if (output.description) {
                        output.description = marked(output.description);
                    }
                }
            }
            if (schema.references) {
                for (const reference of schema.references) {
                    if (reference) {
                        reference.description = marked(reference.description);
                    }
                }
            }
            if (schema.note) {
                schema.note = marked(schema.note);
            }
            if (schema.example) {
                schema.example = marked(schema.example);
            }
            return schema;
        }
        return '';
    }

    /// [Deprecated]
    get category() {
        const schema = this._metadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : '';
    }

    /// [New in v4]
    get metadata() {
        return this._metadata.type(this._operator);
    }

    get attributes() {
        return this._attributes;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get chain() {
        return this._chain;
    }

    set chain(v) {
        this._chain = v;
    }
};

tennis.Attribute = class {
    /**
     * 
     * @param {tennis.Metadata} metadata 
     * @param {string} operator 
     * @param {string} name 
     * @param {utils.Tensor} value 
     */
    constructor(metadata, operator, name, value) {
        this._name = name;
        this._value = value.value;
        if (this._value === null) {
            this._value = "Not readable.";
        }
        this._show_value = null;
        this._type = value.proto;
        this._description = null;
        const schema = metadata.getAttributeSchema(operator, name);
        if (schema) {
            this._description = schema.description;
            // update value, based on schema type
            switch (schema.type) {
                case "bool":
                case "boolean":
                    this._show_value = this._value ? true : false;
                    this._type = "bool";
                    break;
                case "enum":
                    this._type = "enum";
                    try {
                        this._show_value = schema.enum[this._value];
                    }
                    catch (error) {
                    }
                    break;
            }
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                if (this.value == schema.default ||
                    this.value.toString() == schema.default.toString()) {
                    this._visible = false;
                }
            }
            if (this._type == "enum") {
                // show enum as value : string
                this._show_value = this._value + " : " + this._show_value;
            }
        }
    }

    get description() {
        return this._description;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        if (this._show_value === null) {
            return this._value;
        }
        return this._show_value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

tennis.Tensor = class {

    /**
     * 
     * @param {utils.Tensor} tensor 
     */
    constructor(tensor, name=null) {
        this._type = new tennis.TensorType(
            utils.dtype.type_str(tensor.dtype),
            new tennis.TensorShape(tensor.shape));
        // this._value = tensor.value;  // do not decode by default
        this._name = name || '';
        this._tensor = tensor;
    }

    get kind() {
        return 'Constant';
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get state() {
        if (this._tensor.viewable)
        {
            return null;
        }
        return "Tensor data is not readable."
    }

    get value() {
        return this._tensor.view();
    }

    _int_array_string() {
        let str = "[";
        return str;
    }

    toString() {
        const limit = 1000;
        const value = this._tensor.view(limit);
        if (value === null) {
            return "";
        }
        if (false ||
                this._tensor.dtype == utils.dtype.INT8 ||
                this._tensor.dtype == utils.dtype.INT16 ||
                this._tensor.dtype == utils.dtype.INT32 ||
                this._tensor.dtype == utils.dtype.INT64 ||
                this._tensor.dtype == utils.dtype.UINT8 ||
                this._tensor.dtype == utils.dtype.UINT16 ||
                this._tensor.dtype == utils.dtype.UINT32 ||
                this._tensor.dtype == utils.dtype.UINT64) {
            if (this._tensor.shape.length == 1 && this._tensor.shape[0] <= limit) {
                return JSON.stringify(value, null); // parse to single line
            }
        }
        return JSON.stringify(value, null, 2);
    }
};

tennis.TensorType = class {
    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return (this._dataType || '?') + this._shape.toString();
    }
};

tennis.TensorShape = class {

    /**
     * 
     * @param {utils.Tensor} tensor 
     */
    constructor(dimensions) {
        if (dimensions.some((dimension) => dimension === 0 || dimension === undefined || isNaN(dimension))) {
            throw new tennis.Error('Invalid tensor shape.');
        }
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (this._dimensions) {
            if (this._dimensions.length == 0) {
                return '';
            }
            return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
        }
        return '';
    }
};

tennis.Metadata = class {

    static open(host) {
        if (tennis.Metadata._metadata) {
            return Promise.resolve(tennis.Metadata._metadata);
        }
        return host.request(null, 'tennis-metadata.json', 'utf-8').then((data) => {
            tennis.Metadata._metadata = new tennis.Metadata(data);
            return tennis.Metadata._metadata;
        }).catch(() => {
            tennis.Metadata._metadata = new tennis.Metadata(null);
            return tennis.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        this._attributeMap = new Map();
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item && item.name && item.schema) {
                        if (this._map.has(item.name)) {
                            throw new tennis.Error("Duplicate metadata key '" + item.name + "'.");
                        }
                        item.schema.name = item.name;
                        this._map.set(item.name, item.schema);
                    }
                }
            }
        }
    }

    type(operator) {
        return this.getSchema(operator);
    }

    attribute(operator, name) {
        return this.getAttributeSchema(operator, name);
    }

    getSchema(operator) {
        return this._map.get(operator) || null;
    }

    getAttributeSchema(operator, name) {
        const key = operator + ':' + name;
        if (!this._attributeMap.has(key)) {
            this._attributeMap.set(key, null);
            const schema = this.getSchema(operator);
            if (schema && schema.attributes) {
                for (const attribute of schema.attributes) {
                    this._attributeMap.set(operator + ':' + attribute.name, attribute);
                }
            }
        }
        return this._attributeMap.get(key);
    }
};

tennis.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading tennis model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = tennis.ModelFactory;
    module.exports.utils = utils;
}
