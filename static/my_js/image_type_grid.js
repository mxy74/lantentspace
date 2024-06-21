/*
    这个js的功能主要是绘制类别网格
*/
//图片类别颜色
// 师兄原版颜色
// var labels_colors = { 0: '#8dd3c7', 1: '#ffffb3', 2: '#bebada', 3: '#fb8072', 4: '#80b1d3', 5: '#fdb462', 6: '#b3de69', 7: '#fccde5', 8: '#d9d9d9', 9: '#bc80bd', 10: "white"};
// colorbrewer颜色
// var labels_colors = { 0: '#a6cee3', 1: '#1f78b4', 2: '#b2df8a', 3: '#33a02c', 4: '#fb9a99', 5: '#e31a1c', 6: '#fdbf6f', 7: '#ff7f00', 8: '#cab2d6', 9: '#6a3d9a', 10: "white" };
// var steeringAngle_labels_colors = { 0: '#a6cee3', 1: '#1f78b4', 2: '#b2df8a', 3: '#33a02c', 4: '#fb9a99', 5: '#e31a1c', 6: '#fdbf6f', 7: '#ff7f00', 8: '#cab2d6', 9: '#6a3d9a', 10: "white"};
var labels_colors = { 0: '#1f76b3', 1: '#adc5e6', 2: '#fd7e0e', 3: '#fdba77',
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

//CIFAR10图片类别名字
const cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
//SteeringAngle图片类别名字
const SteeringAngle_labels = ['-80~-64', '-64~-48', '-48~-32', '-32~-16', '-16~0', '0~16', '16~32', '32~48', '48~64', '64~80'];
const GTSRB_classes = [
  '20_speed',
  '30_speed',
  '50_speed',
  '60_speed',
  '70_speed',
  '80_speed',
  '80_lifted',
  '100_speed',
  '120_speed',
  'no_overtaking_general',
  'no_overtaking_trucks',
  'right_of_way_crossing',
  'right_of_way_general',
  'give_way',
  'stop',
  'no_way_general',
  'no_way_trucks',
  'no_way_one_way',
  'attention_general',
  'attention_left_turn',
  'attention_right_turn',
  'attention_curvy',
  'attention_bumpers',
  'attention_slippery',
  'attention_bottleneck',
  'attention_construction',
  'attention_traffic_light',
  'attention_pedestrian',
  'attention_children',
  'attention_bikes',
  'attention_snowflake',
  'attention_deer',
  'lifted_general',
  'turn_right',
  'turn_left',
  'turn_straight',
  'turn_straight_right',
  'turn_straight_left',
  'turn_right_down',
  'turn_left_down',
  'turn_circle',
  'lifted_no_overtaking_general',
  'lifted_no_overtaking_trucks'
];

//修改右边的网格类别概览
function change_imageTypeGrid() {
    // var imageTypeGrid_svg = d3.select("#imageTypeGrid_svg");
    // imageTyprGrid_svg.selectAll("rect").remove(); //先删除现有的
    draw_imageTypeGrid();
}


// 显示类别网格
function draw_imageTypeGrid() {
    var svg_typeGrid = d3.select("#imageTypeGrid_svg")
        .attr("width", imageGrid_width)
        .attr("height", imageGrid_height)
    var svg_typeGrid_right = d3.select("#imageTypeGrid_svg_right")
        .attr("width", imageGrid_width)
        .attr("height", imageGrid_height)

    var rectWidth = 40


    // 显示单选按钮选择的模型预测类别
    const selectedRadioModelButton = document.querySelector('input[name="radio_model"]:checked');
    const selectedValue = selectedRadioModelButton.value;
    var img_labels_lst_400;
    var img_fore_labels_lst_400;
    if (selectedValue != "M1-M2") {
        img_labels_lst_400 = img_DNN_output_lst_400_dict[selectedValue]
        img_fore_labels_lst_400 = img_DNN_for_output_lst_400_dict[selectedValue]
    } else {
        var type1 = img_DNN_output_lst_400_dict["M1"]
        var type2 = img_DNN_output_lst_400_dict["M2"]
        var type3 = img_DNN_for_output_lst_400_dict["M1"]
        var type4 = img_DNN_for_output_lst_400_dict["M2"]
        img_labels_lst_400 = type1.map((val, index) => parseInt(val) != parseInt(type2[index]) ? val : 10)
        img_fore_labels_lst_400 = type3.map((val, index) => parseInt(val) != parseInt(type4[index]) ? val : 10)

    }

    // 不同的数据集对应不同的数据操作
    var dataset_type = document.getElementById("select_dataset_type_selection").value //在Document.querySelector()通过id获取才需要加#
    console.log("dataset_type: ", dataset_type)
    if (dataset_type == "CIFAR10") {
        text_list = cifar10_classes
        labels_colors = { 0: '#a6cee3', 1: '#1f78b4', 2: '#b2df8a', 3: '#33a02c', 4: '#fb9a99', 5: '#e31a1c', 6: '#fdbf6f', 7: '#ff7f00', 8: '#cab2d6', 9: '#6a3d9a' };
         const container = document.getElementById('types');

        // 清空已有的颜色块和类别名称
        const colorBlock_no_text = document.querySelectorAll('.color_block_no_text');
        const colorBlock = document.querySelectorAll('.color_block');
        const  classNameDiv = document.querySelectorAll('.classes_name_div');

        // 删除所有 .color_block 和 .classes_name_div 元素
        colorBlock_no_text.forEach(element => element.remove());
        colorBlock.forEach(element => element.remove());
        classNameDiv.forEach(element => element.remove());
        // 默认10个，其余的动态生成，43个颜色块和类别名称
        for (let i = 0; i < cifar10_classes.length; i++) {

          const colorBlock = document.createElement('div');
          colorBlock.className = 'color_block';
          colorBlock.id = `color${i}`;
          // color = labels_colors[i]
          // colorBlocks.style.backgroundColor = color;

          const classNameDiv = document.createElement('div');
          classNameDiv.className = 'classes_name_div';

          const classNameP = document.createElement('p');
          classNameP.className = 'classes_name';
          classNameP.textContent = GTSRB_classes[i];

          classNameDiv.appendChild(classNameP);
          container.appendChild(colorBlock);
          container.appendChild(classNameDiv);
        }
        const colorBlocks = document.querySelectorAll('.color_block');
        for (let i = 0; i < colorBlocks.length; i++) {
            const colorIndex = i % Object.keys(labels_colors).length;
            const color = labels_colors[colorIndex];
            colorBlocks[i].style.backgroundColor = color;
        }
    } else if(dataset_type == "GTSRB"){



        labels_colors = {
            0: '#1f76b3', 1: '#adc5e6', 2: '#fd7e0e', 3: '#fdba77',
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
        const container = document.getElementById('types');

        // 清空已有的颜色块和类别名称
        const colorBlock = document.querySelectorAll('.color_block');
        const colorBlock_no_text = document.querySelectorAll('.color_block_no_text');
        const classNameDiv = document.querySelectorAll('.classes_name_div');

        // 删除所有 .color_block 和 .classes_name_div 元素
        colorBlock.forEach(element => element.remove());
        classNameDiv.forEach(element => element.remove());
        colorBlock_no_text.forEach(element => element.remove());

        // 默认10个，其余的动态生成，43个颜色块和类别名称
        for (let i = 0; i < GTSRB_classes.length; i++) {

          const colorBlock = document.createElement('div');
          colorBlock.className = 'color_block_no_text';
          colorBlock.id = `color${i}`;
          // color = labels_colors[i]
          // colorBlocks.style.backgroundColor = color;
          colorBlock.title = GTSRB_classes[i];  // 设置悬浮提示,悬浮定义在css文件中。
          //  const classNameDiv = document.createElement('div');
          // classNameDiv.className = 'classes_name_div';
          //
          // const classNameP = document.createElement('p');
          // classNameP.className = 'classes_name';
          // classNameP.textContent = GTSRB_classes[i];

          // classNameDiv.appendChild(classNameP);
          container.appendChild(colorBlock);
          // container.appendChild(classNameDiv);
        }
         text_list = GTSRB_classes
        // labels_colors = { 0: '#a6cee3', 1: '#1f78b4', 2: '#b2df8a', 3: '#33a02c', 4: '#fb9a99', 5: '#e31a1c', 6: '#fdbf6f', 7: '#ff7f00', 8: '#cab2d6', 9: '#6a3d9a' };
        const colorBlocks = document.querySelectorAll('.color_block_no_text');
        for (let i = 0; i < colorBlocks.length; i++) {
            // console.log('Object.keys(labels_colors).length;',Object.keys(labels_colors).length, colorBlocks.length)
            // const colorIndex = i % Object.keys(labels_colors).length;
            const color = labels_colors[i];
            colorBlocks[i].style.backgroundColor = color;
        }
    }else if (dataset_type == "SteeringAngle") {
        text_list = SteeringAngle_labels
        labels_colors = { 0: '#543005', 1: '#8c510a', 2: '#bf812d', 3: '#dfc27d', 4: '#f6e8c3', 5: '#c7eae5', 6: '#80cdc1', 7: '#35978f', 8: '#01665e', 9: '#003c30' }
        const colorBlocks = document.querySelectorAll('.color_block');
        for (let i = 0; i < colorBlocks.length; i++) {
            const colorIndex = i % Object.keys(labels_colors).length;
            const color = labels_colors[colorIndex];
            colorBlocks[i].style.backgroundColor = color;
        }
        // console.log("img_labels_lst_400.length", img_labels_lst_400.length)
        // console.log("变换前img_labels_lst_400: ", img_labels_lst_400)
        // 从0~1变换到-80~80
        for (let index = 0; index < img_labels_lst_400.length; index++) {
            img_labels_lst_400[index] = (img_labels_lst_400[index] - 0.5) * 160
        }
        for (let index = 0; index < img_fore_labels_lst_400.length; index++) {
            img_fore_labels_lst_400[index] = (img_fore_labels_lst_400[index] - 0.5) * 160
        }
        // console.log("变换中img_labels_lst_400: ", img_labels_lst_400)
        // 对连续的值进行分箱处理
        for (let index = 0; index < img_labels_lst_400.length; index++) {
            for (let i = 0; i < text_list.length; i++) {
                const range = text_list[i].split('~').map(Number);
                if (img_labels_lst_400[index] >= range[0] && img_labels_lst_400[index] < range[1]) {
                    img_labels_lst_400[index] = i;
                    break;
                }
            }
        }
        for (let index = 0; index < img_fore_labels_lst_400.length; index++) {
            for (let i = 0; i < text_list.length; i++) {
                const range = text_list[i].split('~').map(Number);
                if (img_fore_labels_lst_400[index] >= range[0] && img_fore_labels_lst_400[index] < range[1]) {
                    img_fore_labels_lst_400[index] = i;
                    break;
                }
            }
        }
        // console.log("变换后img_labels_lst_400: ", img_labels_lst_400)
    }

    svg_typeGrid.select("#imageTypeGrid_g")
        .selectAll("rect")
        .data(img_labels_lst_400)
        .join("rect")
        .attr("number", (d, i) => i) //方便后续定位图片
        .attr("class", "class_rect")
        .attr("fill", function (d, i) {
            return labels_colors[parseInt(d)];
        })
        .attr("stroke", "black")
        .attr("height", rectWidth)
        .attr("width", rectWidth)
        .attr("x", (d, i) => rectWidth * (i % 20))
        .attr("y", (d, i) => rectWidth * Math.floor(i / 20))
    svg_typeGrid_right.select("#imageTypeGrid_g_right")
        .selectAll("rect")
        .data(img_fore_labels_lst_400)
        .join("rect")
        .attr("number", (d, i) => i) //方便后续定位图片
        .attr("class", "class_rect")
        .attr("fill", function (d, i) {
            return labels_colors[parseInt(d)];
        })
        .attr("stroke", "black")
        .attr("height", rectWidth)
        .attr("width", rectWidth)
        .attr("x", (d, i) => rectWidth * (i % 20))
        .attr("y", (d, i) => rectWidth * Math.floor(i / 20))

    //给标签上名字(右边部分)
    d3.select("#types_right")
        .selectAll(".classes_name")
        .each(function (d, i) {
            this.innerHTML = text_list[i];
        });
    d3.select("#types")
        .selectAll(".classes_name")
        .each(function (d, i) {
            this.innerHTML = text_list[i];
        });

}
