$(document).ready(function() { 
  $('form').on('submit', function(event) {
    //
    $.ajax({
      data : {
        name : $('#nameInput').val().toLowerCase()
      },
      type : 'POST',
      url : '/similardoc'
    })
    .done(function(data) {
      if (data.error) {
        $('#errorAlert').text(data.error).show();
      }
      else {
        $('.collection').empty();
        $('#searchAlert').text(data.search).show();
        $('#errorAlert').hide();
         for(var i=0; i<data.name.length; i++){
             var perfume = data.name[i].toString().split(",")[0]
             var percent = data.name[i].toString().split(",")[1]
             var rounded = Math.round(parseInt(percent * 100))
            $('.collection').append('<li class="collection-item className"><h4>' + perfume + '</h4>'
              + '<h4 class="right-align">' + rounded + '% match' + '</h4></li>')
            /*for(var i=0; i<data.doc.length; i++){
              $('.collection').append('</h4>' + '<p>' + doc[i].title + '<p>')
            }*/
         }
      }
    });
    event.preventDefault();
  });
});