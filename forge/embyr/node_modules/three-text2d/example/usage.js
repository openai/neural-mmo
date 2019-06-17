var THREE = require('three');
var OrbitControls = require('three-orbit-controls')(THREE);

var THREE_Text = require('../src/index');

var WIDTH = window.innerWidth , HEIGHT = window.innerHeight

var MeshText2D = THREE_Text.MeshText2D;
var SpriteText2D = THREE_Text.SpriteText2D;
var textAlign = THREE_Text.textAlign;

var raycaster = new THREE.Raycaster();
var mouse = new THREE.Vector2();

function Application (container) {
  this.container = container;

  this.camera = new THREE.PerspectiveCamera(60, WIDTH / HEIGHT, 1, 5000),
  this.camera.position.set(0, 0, 500)

  this.controls = new OrbitControls(this.camera)

  this.renderer = new THREE.WebGLRenderer({
    alpha: true,
    antialias: true,
  })
  this.renderer.setSize(WIDTH, HEIGHT)
  this.renderer.setClearColor(0xffff00)
  this.container.appendChild(this.renderer.domElement)

  this.scene = new THREE.Scene();

  this.text = new MeshText2D("CENTER", {
    align: textAlign.center,
    font: '30px Arial',
    fillStyle: '#000000',
    shadowColor: 'rgba(0, 0, 0, 0.2)',
    shadowBlur: 3,
    shadowOffsetX: 2,
    shadowOffsetY: 2
  })
  this.text.material.alphaTest = 0.1
  this.text.position.set(0,0,0)
  this.text.scale.set(1.5,1.5,1.5)
  this.scene.add(this.text)

  this.text2 = new MeshText2D("LEFT", { align: textAlign.left, font: '30px Arial', fillStyle: '#000000' })
  this.text2.material.alphaTest = 0.1
  this.text2.position.set(0,100,0)
  this.text2.scale.set(1.5,1.5,1.5)
  this.scene.add(this.text2)

  this.text3 = new MeshText2D("RIGHT", { align: textAlign.right, font: '30px Arial', fillStyle: '#000000' })
  this.text3.material.alphaTest = 0.1
  this.text3.position.set(0,-100,0)
  this.text3.scale.set(1.5,1.5,1.5)
  this.scene.add(this.text3)

  this.sprite = new SpriteText2D("SPRITE", { align: textAlign.center, font: '30px Arial', fillStyle: '#000000'})
  this.sprite.position.set(0, -200, 0)
  this.sprite.scale.set(1.5, 1.5, 1.5)
  this.sprite.material.alphaTest = 0.1
  this.scene.add(this.sprite)

  var i = 0
  setInterval(() => {
    this.text.text = "CENTER" + i
    this.text2.text = "LEFT" + i
    this.text3.text = "RIGHT" + i
    this.sprite.text = "SPRITE " + i
    i++
  }, 50)

  window.addEventListener('resize', this.onResize.bind(this), false)
  window.addEventListener('mousemove', this.onMouseMove.bind(this), false)
}

Application.prototype.onResize = function (e) {
  WIDTH = window.innerWidth
  HEIGHT = window.innerHeight

  this.renderer.setSize(WIDTH, HEIGHT)
  this.camera.aspect = WIDTH / HEIGHT
  this.camera.updateProjectionMatrix()
}

Application.prototype.onMouseMove = function (event) {
  mouse.x = ( event.clientX / window.innerWidth  ) * 2 - 1;
  mouse.y = - ( event.clientY / window.innerHeight  ) * 2 + 1;
}

Application.prototype.loop = function () {
  this.text.rotation.y += 0.010
  this.text2.rotation.y += 0.015
  this.text3.rotation.y += 0.02

  // update the picking ray with the camera and mouse position
  raycaster.setFromCamera( mouse, this.camera );

  // calculate objects intersecting the picking ray
  var intersects = raycaster.intersectObjects( this.scene.children );

  for ( var i = 0; i < intersects.length; i++ ) {
    console.log("found:", intersects[ i ].object)
  }

  this.renderer.render(this.scene, this.camera)
  requestAnimationFrame(this.loop.bind(this))
}

var app = new Application(document.body)
app.loop()
