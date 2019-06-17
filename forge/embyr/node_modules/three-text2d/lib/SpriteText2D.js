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
var Text2D_1 = require("./Text2D");
var SpriteText2D = (function (_super) {
    __extends(SpriteText2D, _super);
    function SpriteText2D() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    SpriteText2D.prototype.raycast = function () {
        return this.sprite.raycast.apply(this.sprite, arguments);
    };
    SpriteText2D.prototype.updateText = function () {
        this.canvas.drawText(this._text, {
            font: this._font,
            fillStyle: this._fillStyle
        });
        // cleanup previous texture
        this.cleanUp();
        this.texture = new THREE.Texture(this.canvas.canvas);
        this.texture.needsUpdate = true;
        this.applyAntiAlias();
        if (!this.material) {
            this.material = new THREE.SpriteMaterial({ map: this.texture });
        }
        else {
            this.material.map = this.texture;
        }
        if (!this.sprite) {
            this.sprite = new THREE.Sprite(this.material);
            this.geometry = this.sprite.geometry;
            this.add(this.sprite);
        }
        this.sprite.scale.set(this.canvas.width, this.canvas.height, 1);
        this.sprite.position.x = ((this.canvas.width / 2) - (this.canvas.textWidth / 2)) + ((this.canvas.textWidth / 2) * this.align.x);
        this.sprite.position.y = (-this.canvas.height / 2) + ((this.canvas.textHeight / 2) * this.align.y);
    };
    return SpriteText2D;
}(Text2D_1.Text2D));
exports.SpriteText2D = SpriteText2D;
