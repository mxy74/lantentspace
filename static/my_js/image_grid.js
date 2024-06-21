/*
    这个js主要用来绘制右边的网格图片概览
*/

//修改右边的网格图片概览
function change_imageGrid() {
    var imageGrid_svg = d3.select("#imageGrid_svg");
    imageGrid_svg.selectAll("image").remove(); //先删除现有的
    var imageGrid_svg_right = d3.select("#imageGrid_svg_right");
    imageGrid_svg_right.selectAll("image").remove(); //先删除现有的
    draw_imageGrid();
}

//绘制右边图片概览
function draw_imageGrid() {
    // 停止转动加载图标
    var spinner = d3.selectAll(".spinner-border");
    spinner.style("display", "none");

    var svg_imageGrid = d3.select("#imageGrid_svg")
        .attr("width", imageGrid_width)
        .attr("height", imageGrid_height)
    var svg_imageGrid_right = d3.select("#imageGrid_svg_right")
        .attr("width", imageGrid_width)
        .attr("height", imageGrid_height)

    var rectWidth = 40

    var svg_imageGrid_g = svg_imageGrid.select("#imageGrid_g");
    var svg_imageGrid_g_right = svg_imageGrid_right.select("#imageGrid_g_right");
    // console.log("image_grid.js中img_coords_lst_400_dict: ", img_coords_lst_400_dict)
    // console.log("image_grid.js中robustness_dict: ", robustness_dict)

    var img_fore_labels_lst_400 = img_DNN_for_output_lst_400_dict["M1"]
    console.log(img_fore_labels_lst_400)
    labels_colors = { 0: '#1f76b3', 1: '#adc5e6', 2: '#fd7e0e', 3: '#fdba77',
            4: '#2c9f2c', 5: '#97dd89', 6: '#d42728', 7: '#fd9795',
            8: '#9366bc', 9: '#c3afd3', 10: "#8b554a", 11: '#c29b93',
            12: '#e176c0', 13: '#f5b5d0', 14: '#7e7e7e', 15: '#c5c5c5',
            16: '#bbbc22', 17: '#d9d98c', 18: '#17bdcd', 19: '#9dd8e3',
            20: '#393b78', 21: "#627839" , 22: '#8b6c31', 23: '#833c39',
            24: '#7a4072', 25: '#fdeb6e', 26: '#cae9c3', 27: '#f97f71',
            28: '#f9b3ad', 29: '#e52989', 30: '#be5a16', 31: '#502203' ,
            32: '#023d64', 33: '#1a3d08', 34: '#576200', 35: '#490302',
            36: '#300067', 37: '#55347e', 38: '#855b2f', 39: '#86005a',
            40: '#490150', 41: "#30e5bb",42: '#025d41'};
    var img_coords_lst_400 = img_coords_lst_400_dict["M1"]
    //添加边框
    svg_imageGrid_g.selectAll("rect")
        .data(img_coords_lst_400)
        .join("rect")
        .attr("class", "img_border_rect")
        .attr("fill", "none")
        .attr("stroke", "black")
        .attr("stroke-width", "1")
        .attr("height", rectWidth)
        .attr("width", rectWidth)
        .attr("x", (d, i) => rectWidth * (i % 20))
        .attr("y", (d, i) => rectWidth * Math.floor(i / 20));
    svg_imageGrid_g_right.selectAll("rect")
        .data(img_coords_lst_400)
        .join("rect")
        .attr("class", "img_border_rect")
        .attr("fill", (d, i) => labels_colors[parseInt(img_fore_labels_lst_400[i])])
        .attr("stroke", "white")
        .attr("stroke-width", "0.5")
        // .attr("stroke", (d, i) => labels_colors[img_fore_labels_lst_400[i]]) // 根据类别信息设置边框颜色
        // .attr("stroke-width", "3") // 设置边框宽度为2像素
        .attr("height", rectWidth)
        .attr("width", rectWidth)
        .attr("x", (d, i) => rectWidth * (i % 20))
        .attr("y", (d, i) => rectWidth * Math.floor(i / 20));

    var dataset_type = document.getElementById("select_dataset_type_selection").value
    //添加图片
    svg_imageGrid_g.selectAll("image")
        .data(img_coords_lst_400)
        .join("image")
        .attr("number", (d, i) => i) //方便后续定位图片
        .attr("href", (d, i) => `../static/data/${dataset_type}/pic/grid_images/grid_image_${i}.png?t=` + Math.random())
        .attr("height", rectWidth)
        .attr("width", rectWidth)
        .attr("x", (d, i) => rectWidth * (i % 20))
        .attr("y", (d, i) => rectWidth * Math.floor(i / 20))
        .attr("data-bs-toggle", "tooltip")
        .attr("title", "image")
        .attr("border", "5")
    svg_imageGrid_g_right.selectAll("image")
        .data(img_coords_lst_400)
        .join("image")
        .attr("number", (d, i) => i) //方便后续定位图片
        .attr("href", (d, i) => `../static/data/${dataset_type}/pic/grid_fore_images/grid_fore_image_${i}.png?t=` + Math.random())
        .attr("height", rectWidth-6)
        .attr("width", rectWidth-6)
        .attr("x", (d, i) => rectWidth * (i % 20)+3)
        .attr("y", (d, i) => rectWidth * Math.floor(i / 20)+3)
        .attr("data-bs-toggle", "tooltip")
        .attr("title", "image")
        .attr("border", "5")
            // 添加CAM图像层

    const camImages = svg_imageGrid_g_right.selectAll(".cam-image")
        .data(img_coords_lst_400)
        .join("image")
        // .attr("class", "cam-image")
        .attr("number", (d, i) => i) // 方便后续定位图片
        .attr("href", (d, i) => `../static/data/${dataset_type}/pic/cam_image/cam_image_${i}.png?t=` + Math.random())
        .attr("height", rectWidth - 6)
        .attr("width", rectWidth - 6)
        .attr("x", (d, i) => rectWidth * (i % 20) + 3)
        .attr("y", (d, i) => rectWidth * Math.floor(i / 20) + 3)
        .attr("data-bs-toggle", "tooltip")
        .attr("title", "CAM image")
        // .attr("opacity", 0.5); // 设置透明度以便原始图像可见
    const camBtn = document.querySelector("#radio_cam");
    camBtn.addEventListener("click", function () {
        const currentOpacity = camImages.attr("opacity");
        const newOpacity = currentOpacity === "0" ? "1" : "0";
        camImages.attr("opacity", newOpacity);
     })
}
