"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var THREE = require("three");
var utils_1 = require("./utils");
var CanvasText_1 = require("./CanvasText");
var Text2D = (function (_super) {
    __extends(Text2D, _super);
    function Text2D(text, options) {
        if (text === void 0) { text = ''; }
        if (options === void 0) { options = {}; }
        var _this = _super.call(this) || this;
        _this._font = options.font || '30px Arial';
        _this._fillStyle = options.fillStyle || '#FFFFFF';
        _this._shadowColor = options.shadowColor || 'rgba(0, 0, 0, 0)';
        _this._shadowBlur = options.shadowBlur || 0;
        _this._shadowOffsetX = options.shadowOffsetX || 0;
        _this._shadowOffsetY = options.shadowOffsetY || 0;
        _this.canvas = new CanvasText_1.CanvasText();
        _this.align = options.align || utils_1.textAlign.center;
        _this.side = options.side || THREE.DoubleSide;
        // this.anchor = Label.fontAlignAnchor[ this._textAlign ]
        _this.antialias = (typeof options.antialias === "undefined") ? true : options.antialias;
        _this.text = text;
        return _this;
    }
    Object.defineProperty(Text2D.prototype, "width", {
        get: function () { return this.canvas.textWidth; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(Text2D.prototype, "height", {
        get: function () { return this.canvas.textHeight; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(Text2D.prototype, "text", {
        get: function () { return this._text; },
        set: function (value) {
            if (this._text !== value) {
                this._text = value;
                this.updateText();
            }
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(Text2D.prototype, "font", {
        get: function () { return this._font; },
        set: function (value) {
            if (this._font !== value) {
                this._font = value;
                this.updateText();
            }
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(Text2D.prototype, "fillStyle", {
        get: function () {
            return this._fillStyle;
        },
        set: function (value) {
            if (this._fillStyle !== value) {
                this._fillStyle = value;
                this.updateText();
            }
        },
        enumerable: true,
        configurable: true
    });
    Text2D.prototype.cleanUp = function () {
        if (this.texture) {
            this.texture.dispose();
        }
    };
    Text2D.prototype.applyAntiAlias = function () {
        if (this.antialias === false) {
            this.texture.magFilter = THREE.NearestFilter;
            this.texture.minFilter = THREE.LinearMipMapLinearFilter;
        }
    };
    return Text2D;
}(THREE.Object3D));
exports.Text2D = Text2D;
