export {makeTextSprite};


function makeTextSprite(message, fontsize, color) {
    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');
    ctx.font = fontsize + "px DragonSlapper";

    if (color == 'undefined') {
       color = '#' + (Math.random()*0xFFFFFF<<0).toString(16);
    }

    // setting canvas width/height before ctx draw, else canvas is empty
    canvas.width = ctx.measureText(message).width;
    canvas.height = fontsize * 1.5;

    // after setting the canvas width/height we have to re-set font to apply
    // looks like ctx reset
    ctx.font = fontsize + "px DragonSlapper";
    ctx.fillStyle = color;
    ctx.fillText(message, 0, fontsize);

    var texture = new THREE.Texture(canvas);
    texture.minFilter = THREE.NearestFilter; // NearestFilter;
    texture.needsUpdate = true;

    var spriteMaterial = new THREE.SpriteMaterial({
       map : texture, sizeAttenuation : false});
    var sprite = new THREE.Sprite(spriteMaterial);
    sprite.scale.set(0.1, 0.03, 1);
    return sprite;
}
