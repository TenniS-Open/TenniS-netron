/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */
/* TenniS: Tensor based Edge Neural Network Inference System */

var tennis = tennis || {};
var base = base || require('./base');
var long = long || { Long: require('long') };
var marked = marked || require('marked');
var ts = ts || require("./tennis-utils")

tennis.ModelFactory = class {

    match(context) {
        // const extension = context.identifier.split('.').pop().toLowerCase();
        const b = context.buffer;
        let stream = new ts.Stream(b)
        stream.skip(4)
        const mask = stream.int32()
        return mask == 0x19910929;
    }

    open(context, host) {
        return tennis.Metadata.open(host).then((metadata) => {
            const identifier = context.identifier;
            let stream = new ts.Stream(context.buffer);
            return this._openModel(metadata, stream, identifier);
        });
    }
    _openModel(metadata, stream, identifier) {
        try {
            return new tennis.Model(metadata, stream);
        }
        catch (error) {
            let message = error && error.message ? error.message : error.toString();
            message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
            throw new tennis.Error(message + " in '" + identifier + "'.");
        }
    }
};

tennis.Model = class {

    constructor(metadata, stream) {
        this._graphs = [];
        this._graphs.push(new tennis.Graph(metadata, stream));
    }

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

        let graph = new ts.Module(stream);

        for (const input of graph.inputs) {
            this._inputs.push(new tennis.Parameter(input.name, true, [new tennis.Argument(input.arg_id, "float32:[1,2,3]", "", null)]))
        }
        for (const output of graph.outputs) {
            this._outputs.push(new tennis.Parameter(output.name, true, [new tennis.Argument(output.arg_id, "", "", null)]))
        }

        for (const node of graph.nodes) {
            if (node.op == "<param>") continue;
            if (node.op == "<const>") continue;
            this._nodes.push(new tennis.Node(metadata, node))
        }
    }

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
    constructor(id, type, description, initializer) {
        this._id = id;
        this._type = type || null;
        this._description = description || '';
        this._initializer = initializer || null;
    }

    get id() {
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
     * @param {ts.Node} node 
     */
    constructor(metadata, node) {
        this._name = node.name;
        this._metadata = metadata;
        this._operator = node.op;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._chain = [];

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
                    name: node.inputs.length > 1 ? "input" : "inputs",
                    type: "Tensor",
                    description: "",
                })
            }
            schema_outputs.push({
                name: "output",
                type: "Tensor",
                description: "",
            })
        }
        if (schema_inputs.length == 1 && node.inputs.length > 1) {
            const input = schema_inputs[0];
            let args = [];
            for (let i = 0; i < node.inputs.length; ++i) {
                args.push(new tennis.Argument(node.input(i).arg_id, null, null, null));
            }
            this._inputs.push(new tennis.Parameter(input.name, true, args));
        } else {
            for (let i = 0; i < node.inputs.length; ++i) {
                let input = {name: "output", type: "Tensor", description: ""};
                if (i < schema_inputs.length) {
                    input = schema_inputs[i];
                }
                this._inputs.push(new tennis.Parameter(input.name, true, [
                    new tennis.Argument(node.input(i).arg_id, input.type, input.description)
                ]));
            }
        }
        for (let i = 0; i < schema_outputs.length; ++i) {
            let output = schema_outputs[i];
            this._outputs.push(new tennis.Parameter(output.name, true, [
                new tennis.Argument(node.arg_id + (i == 0 ? "" : "[" + i + "]"),
                output.type, output.description)
            ]));
        }
    }

    get name() {
        return this._name;
    }

    get operator() {
        return this._operator;
    }

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
            return schema;
        }
        return '';
    }

    get category() {
        const schema = this._metadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : '';
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
};

tennis.Attribute = class {

    constructor(metadata, operator, name, value) {
        this._name = name;
        this._value = value;
        const schema = metadata.getAttributeSchema(operator, name);
        if (schema) {
            this._type = schema.type || '';
            switch (this._type) {
                case 'int32': {
                    this._value = parseInt(this._value, 10);
                    break;
                }
                case 'float32': {
                    this._value = parseFloat(this._value);
                    break;
                }
            }
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                if (this._value == schema.default) {
                    this._visible = false;
                }
            }
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

tennis.Tensor = class {

    constructor(type, data) {
        this._type = type;
        this._data = data;
    }

    get kind() {
        return 'Tensor';
    }

    get name() {
        return '';
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state;
    }

    get value() {
        let context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        let context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        const value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        let context = {};
        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        context.state = null;
        context.position = 0;
        context.count = 0;
        context.dataView = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
        context.dimensions = this.type.shape.dimensions;
        return context;
    }

    _decode(context, dimension) {
        let results = [];
        const size = context.dimensions[dimension];
        if (dimension == context.dimensions.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(context.dataView.getFloat32(context.position, true));
                context.position += 4;
                context.count++;
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        return results;
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

tennis.Weights = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._position = 0;
        const major = this.int32();
        const minor = this.int32();
        const revision = this.int32();
        this._seen = ((major * 10 + minor) >= 2) ? this.int64() : this.int32();
        const transpose = (major > 1000) || (minor > 1000);
        if (transpose) {
            throw new tennis.Error("Unsupported transpose weights file version '" + [ major, minor, revision ].join('.') + "'.");
        }
    }

    int32() {
        const position = this._position;
        this.seek(4);
        return this._dataView.getInt32(position, true);
    }

    int64() {
        let hi = this.int32();
        let lo = this.int32();
        return new long.Long(hi, lo, true).toNumber();
    }

    bytes(length) {
        const position = this._position;
        this.seek(length);
        return this._buffer.subarray(position, this._position);
    }

    seek(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new tennis.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    validate() {
        if (this._position !== this._buffer.length) {
            throw new tennis.Error('Invalid weights size.')
        }
    }
}

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
                        this._map.set(item.name, item.schema);
                    }
                }
            }
        }
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
}
