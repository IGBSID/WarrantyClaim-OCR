function readURL(input) {

  if (input.files && input.files[0]) {

    let index = 1;

    $(input.files).each(function () {

      var reader = new FileReader();

      reader.readAsDataURL(this);
      
      reader.onload = function (e) {
        if(index == 1)
        {
          $(".carousel-inner").append("<div class='item active'><img src='" + e.target.result + "' style='width:400px; height: 400px;'></div>");
          $("#dummy").remove();
        }
        else
        {
          $(".carousel-inner").append("<div class='item'><img src='" + e.target.result + "' style='width:400px; height: 400px;'></div>");
        }
        // console.log(index +"--Length--"+ input.files.length)
        if(index == input.files.length)

        {

          setTimeout(()=>{

            var isImageShow = document.getElementById("imageSection");

            isImageShow.style.display = "block";

            setTimeout(()=>{

              document.getElementsByClassName("carousel-inner").innerHTML = document.getElementsByClassName("carousel-inner").innerHTML ;

            },2000)

          },2000)

        }
       
        $("#isFileData").val(e.target.result);

        index++;

      }
      $('.carousel').carousel()  

    });
  }
  else{
    alert("You can select only 8 images")
  }

}

function readOCR(input) {
  // alert(222)
  if (input.files && input.files[0]) {
      var reader = new FileReader();


      // alert(reader)
     reader.onload = function (e) {
          $('#image')
              .attr('src', e.target.result);
              var isImageShow = document.getElementById("image");
              isImageShow.style.display = "block";
          $("#isFileData").val(e.target.result)
      };
     reader.readAsDataURL(input.files[0]);
  }
}

function classfiy()
{
    var fileData = $("#isFileData").val()
    $.ajax({

        // Our sample url to make request
        url:'/',

        // Type of Request
        type: "POST",
        data: {
          fileData: fileData
        },

        // Function to call when to
        // request is ok
        success: function (data) {
            alert("done")
        },

        // Error handling
        error: function (error) {
            console.log(`Error ${error}`);
        }
    });

}

function cancelRecord()
{
  document.getElementById('dresult').innerHTML='';
  window.location.href = "/home";
}

function cancelTsnRecord()
{
  document.getElementById('extracted_value').innerHTML='';
  window.location.href = "/tsn_home";
}

function cancelTrnRecord()
{
  document.getElementById('extracted_value').innerHTML='';
  window.location.href = "/trn_home";
}

function displayMessage() 
{
  document.getElementById('texto').innerHTML = "Thank You For FeedBack!";
}

function loadExistingFiles() 
{
  // var image_data =  JSON.parse(JSON.stringify((document.getElementById('existing_image').value).toString()));
  var inputs = $(".existing_image");
  for(var i=0;i<inputs.length;i++)
  {
    if(i == 0)
    {
      $(".carousel-inner").append("<div class='item active'><img src='" + $(inputs[i]).val() + "' style='width:400px; height: 400px;'></div>");
      $("#dummy").remove();
    }
    else
    {
      $(".carousel-inner").append("<div class='item'><img src='" + $(inputs[i]).val() + "' style='width:400px; height: 400px;'></div>");
    }

    if(i == inputs.length-1)
    {
      setTimeout(()=>{
        var isImageShow = document.getElementById("imageSection");
        isImageShow.style.display = "block";
        setTimeout(()=>{
          document.getElementsByClassName("carousel-inner").innerHTML = document.getElementsByClassName("carousel-inner").innerHTML ;
        },2000)
      },2000)
      $('.carousel').carousel()
    }
  }
}