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

        let graph = new ts.Module(stream);
        this._mask = graph.mask;

        for (const input of graph.inputs) {
            this._inputs.push(new tennis.Parameter(input.name, true, [new tennis.Argument(input, input.proto, "", null)]))
        }
        for (const output of graph.outputs) {
            this._outputs.push(new tennis.Parameter(output.name, true, [new tennis.Argument(output, output.proto, "", null)]))
        }

        let map_node = {};
        for (const node of graph.nodes) {
            if (node.op == "<param>") continue;
            if (node.op == "<const>") continue;
            let draw_node = new tennis.Node(metadata, node);
            map_node[node] = draw_node;
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
     * @param {ts.Node} node 
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
                    name: "input " + i,
                    type: "tensor",
                    description: "",
                })
            }
            schema_outputs.push({
                name: "output",
                type: "tensor",
                description: "",
            })
        }
        // set inputs which have schema
        let i = null;
        for (i = 0; i < schema_inputs.length; ++i) {
            if (i >= node.inputs.length) break;
            const input = schema_inputs[i];
            if (input.option == "variadic") {
                let args = node.inputs.slice(i, node.inputs.length).map((v, j) => {
                    return new tennis.Argument(node.input(j), null, null, null);
                });
                this._inputs.push(new tennis.Parameter(input.name, true, args));
                i = node.inputs.length;
                break;
            }
            this._inputs.push(new tennis.Parameter(input.name, true, [
                new tennis.Argument(node.input(i), input.type, input.description)
            ]));
        }
        // set inputs with out schema
        for (;i < node.inputs.length; ++i) {
            let input = {name: "input " + i, type: "tensor", description: ""};
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
            let output = {name: "output " + i, type: "tensor", description: ""};
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
     * @param {ts.Tensor} value 
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
     * @param {ts.Tensor} tensor 
     */
    constructor(tensor, name=null) {
        this._type = new tennis.TensorType(
            ts.dtype.type_str(tensor.dtype),
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

    toString() {
        const value = this._tensor.view(1000);
        if (value === null) {
            return "";
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
     * @param {ts.Tensor} tensor 
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
